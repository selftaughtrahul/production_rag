"""
Document management routes.

    POST   /ingest                → Upload and ingest a document
    GET    /documents             → List the authenticated user's documents
    DELETE /documents/{id}        → Delete a document and all its chunks
    GET    /tasks/{task_id}       → Check background task status
"""
from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from typing import List

from app.api.dependencies import build_components, build_ingestion_pipeline
from app.models.schemas import DocumentResponse, UserInDB
from app.services.auth.dependencies import get_current_user
from app.tasks.tasks import ingest_document_task
from database.sqlite import get_db

router = APIRouter(tags=["Documents"])


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _save_upload(file: UploadFile, content: bytes) -> str:
    """Save uploaded bytes to a temp file and return its path."""
    temp_dir = Path("documents/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix

    with NamedTemporaryFile(dir=temp_dir, delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        return tmp.name


# ─────────────────────────────────────────────────────────────
# GET /tasks/{task_id}
# ─────────────────────────────────────────────────────────────

@router.get("/tasks/{task_id}", summary="Check background task status")
async def get_task_status(task_id: str):
    """Get the status and result of a background ingestion task."""
    task_result = AsyncResult(task_id, app=ingest_document_task.app)

    response = {"task_id": task_id, "status": task_result.status}

    if task_result.status == "SUCCESS":
        response["result"] = task_result.result
    elif task_result.status == "FAILURE":
        response["error"] = str(task_result.info)

    return response


# ─────────────────────────────────────────────────────────────
# POST /ingest
# ─────────────────────────────────────────────────────────────

@router.post("/ingest", summary="Upload and ingest a document")
async def ingest_document(
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user),
    db=Depends(get_db),
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
    temp_path = _save_upload(file, content)

    # ── Background mode (Linux/Celery) ─────────────────────
    IS_ASYNC = False  # Set True when Celery worker is running
    if IS_ASYNC:
        task = ingest_document_task.delay(
            source=temp_path,
            document_id=document_id,
            filename=file.filename,
            user_id=current_user.id,
        )
        return {
            "success": True,
            "message": "Upload received. Ingestion started in background.",
            "document_id": document_id,
            "filename": file.filename,
            "task_id": task.id,
            "status": task.status,
        }

    # ── Synchronous mode (dev / Windows) ───────────────────
    pipeline = build_ingestion_pipeline(source=temp_path)
    result = pipeline.ingest(
        source=temp_path,
        document_id=document_id,
        filename=file.filename,
        user_id=current_user.id,
    )

    conn, cursor = db
    cursor.execute(
        "INSERT INTO documents (id, user_id, file_name) VALUES (?, ?, ?)",
        (document_id, current_user.id, file.filename),
    )

    return {
        "success": True,
        "message": "Document ingested successfully",
        "document_id": document_id,
        "filename": file.filename,
        "user_id": current_user.id,
        "ingestion": result,
    }


# ─────────────────────────────────────────────────────────────
# GET /documents
# ─────────────────────────────────────────────────────────────

@router.get("/documents", response_model=List[DocumentResponse], summary="List user documents")
async def list_documents(
    current_user: UserInDB = Depends(get_current_user),
    db=Depends(get_db),
):
    """List all documents uploaded by the authenticated user."""
    conn, cursor = db
    cursor.execute(
        "SELECT id, user_id, file_name, created_at FROM documents WHERE user_id = ?",
        (current_user.id,),
    )
    return [dict(row) for row in cursor.fetchall()]


# ─────────────────────────────────────────────────────────────
# DELETE /documents/{document_id}
# ─────────────────────────────────────────────────────────────

@router.delete("/documents/{document_id}", summary="Delete a document")
async def delete_document(
    document_id: str,
    current_user: UserInDB = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Delete a document and all its vector chunks.
    Only the owning user can delete their own documents.
    """
    if not document_id.strip():
        raise HTTPException(status_code=400, detail="document_id is required")

    components = build_components()
    vector_store = components.vector_store

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

    conn, cursor = db
    cursor.execute(
        "DELETE FROM documents WHERE id = ? AND user_id = ?",
        (document_id, current_user.id),
    )

    return {
        "success": True,
        "message": "Document deleted successfully",
        "document_id": document_id,
        "deleted_chunks": chunks_count,
    }
