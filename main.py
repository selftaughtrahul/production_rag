from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from celery.result import AsyncResult
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from app.api.auth import router as auth_router
from app.api.dependencies import build_components, build_ingestion_pipeline
from app.api.query import router as query_router
from app.models.schemas import UserInDB, DocumentResponse
from app.services.auth.dependencies import get_current_user
from app.tasks.tasks import ingest_document_task
from database.mysql import init_db, get_db

# ─────────────────────────────────────────────────────────────
# Lifespan — runs once on startup
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the MySQL schema on startup."""
    init_db()
    yield


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG API",
    description="Document ingestion and retrieval API with user authentication",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(query_router)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def save_file(file: UploadFile, content: bytes) -> str:
    temp_dir = Path("documents/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix

    with NamedTemporaryFile(
        dir=temp_dir,
        delete=False,
        suffix=suffix,
    ) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name

    return temp_file_path


# ─────────────────────────────────────────────────────────────
# GET /tasks/{task_id}
# ─────────────────────────────────────────────────────────────

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get the status and result of a background ingestion task."""
    task_result = AsyncResult(task_id, app=ingest_document_task.app)

    response = {
        "task_id": task_id,
        "status": task_result.status,
    }

    if task_result.status == "SUCCESS":
        response["result"] = task_result.result
    elif task_result.status == "FAILURE":
        response["error"] = str(task_result.info)

    return response


# ─────────────────────────────────────────────────────────────
# POST /ingest  (protected — requires JWT)
# ─────────────────────────────────────────────────────────────

@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db),
):
    """
    Upload a document and ingest it into the vector store.

    The authenticated user's ID is stored as metadata on every chunk
    so that retrieval can be scoped per-user.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    document_id = str(uuid4())
    original_filename = file.filename
    temp_file_path = save_file(file, content)

    is_linux = False

    if is_linux:
        task = ingest_document_task.delay(
            source=temp_file_path,
            document_id=document_id,
            filename=original_filename,
            user_id=current_user.id,
        )
        return {
            "success": True,
            "message": "Document upload successful. Ingestion started in the background.",
            "document_id": document_id,
            "filename": original_filename,
            "task_id": task.id,
            "status": task.status,
        }

    # ── Synchronous dev mode ──────────────────────────────────
    pipeline = build_ingestion_pipeline(source=temp_file_path)

    result = pipeline.ingest(
        source=temp_file_path,
        document_id=document_id,
        filename=original_filename,
        user_id=current_user.id,        # ← user-scoped metadata
    )

    # Save to SQLite database
    conn, cursor = db
    cursor.execute(
        """
        INSERT INTO documents (id, user_id, file_name)
        VALUES (?, ?, ?)
        """,
        (document_id, current_user.id, original_filename)
    )

    return {
        "success": True,
        "message": "Document ingested successfully",
        "document_id": document_id,
        "filename": original_filename,
        "user_id": current_user.id,
        "ingestion": result,
    }



# ─────────────────────────────────────────────────────────────
# DELETE /documents/{document_id}  (protected — requires JWT)
# ─────────────────────────────────────────────────────────────

@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db),
):
    """
    Delete a document and all its vector chunks.

    Only the owning user can delete their own documents.
    """
    if not document_id.strip():
        raise HTTPException(status_code=400, detail="document_id is required")

    components = build_components()
    vector_store = components.vector_store

    # Verify ownership — count chunks matching both document_id AND user_id
    chunks_count = vector_store.count_document_chunks(
        document_id,
        metadata_filter={"user_id": current_user.id},
    )

    if chunks_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Document not found or you do not have permission to delete it",
        )

    vector_store.delete_document(document_id)

    # Delete from SQLite database
    conn, cursor = db
    cursor.execute(
        "DELETE FROM documents WHERE id = ? AND user_id = ?",
        (document_id, current_user.id)
    )

    return {
        "success": True,
        "message": "Document deleted successfully",
        "document_id": document_id,
        "deleted_chunks": chunks_count,
    }


# ─────────────────────────────────────────────────────────────
# GET /documents  (protected — requires JWT)
# ─────────────────────────────────────────────────────────────

from typing import List

@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db),
):
    """
    List all documents uploaded by the authenticated user.
    """
    conn, cursor = db
    
    cursor.execute(
        "SELECT id, user_id, file_name, created_at FROM documents WHERE user_id = ?",
        (current_user.id,)
    )
    rows = cursor.fetchall()
    
    return [dict(row) for row in rows]
