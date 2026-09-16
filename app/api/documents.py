"""
Document management routes.

 
"""

from pathlib import Path
from uuid import uuid4
from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from typing import List
from app.api.dependencies import build_components, build_ingestion_pipeline
from app.models.schemas import DocumentResponse, UserInDB
from app.services.auth.dependencies import get_current_user
from app.services.retriever.bm25_store import BM25Store
from app.tasks.tasks import ingest_document_task
from database.models import Document
from database.sqlite import get_db
from app.core.config import Settings
from sqlalchemy.orm import Session
from sqlalchemy import select
settings = Settings.from_environment()

router = APIRouter(tags=["Documents"])

_DOCUMENTS_ROOT = Path("documents")


def _save_upload(file: UploadFile,content: bytes,*,user_id: str,document_id: str) -> str:
    """Save the upload under documents/{user_id}/{document_id}{ext}."""
    suffix = Path(file.filename or "upload").suffix
    dest_dir = _DOCUMENTS_ROOT / user_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{document_id}{suffix}"
    dest.write_bytes(content)
    return str(dest)


def _resolve_stored_file(file_path: str | None) -> Path | None:
    """Return a path only if it exists under the documents directory."""
    if not file_path:
        return None

    try:
        resolved = Path(file_path).resolve()
        root = _DOCUMENTS_ROOT.resolve()
        if root not in resolved.parents:
            return None
        return resolved
    except OSError:
        return None


def _delete_stored_file(file_path: str | None) -> None:
    """Remove a stored upload only if it lives under the documents directory."""
    resolved = _resolve_stored_file(file_path)
    if resolved is None:
        return

    try:
        resolved.unlink(missing_ok=True)
    except OSError:
        pass



@router.get("/tasks/{task_id}", summary="Check background task status")
async def get_task_status(task_id: str,current_user: UserInDB = Depends(get_current_user)):
    """Get the status and result of a background ingestion task."""
    task_result = AsyncResult(task_id, app=ingest_document_task.app)

    response = {"task_id": task_id, "status": task_result.status}

    if task_result.status == "SUCCESS":
        result = task_result.result or {}
        if isinstance(result, dict) and result.get("user_id") not in {None, current_user.id}:
            raise HTTPException(status_code=404, detail="Task not found")
        response["result"] = result
    elif task_result.status == "FAILURE":
        response["error"] = "Ingestion failed."

    return response

@router.post("/ingest", summary="Upload and ingest a document")
async def ingest_document(
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db),
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
    file_path = _save_upload(file,content,user_id=current_user.id,document_id=document_id)

    IS_ASYNC = settings.IS_ASYNC
    if IS_ASYNC:
        task = ingest_document_task.delay(
            source=file_path,
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
    pipeline = build_ingestion_pipeline(source=file_path)
    result = pipeline.ingest(
        source=file_path,
        document_id=document_id,
        filename=file.filename,
        user_id=current_user.id,
    )

    db.add(
        Document(
            id=document_id,
            user_id=current_user.id,
            file_name=file.filename,
            file_path=file_path,
        )
    )

    return {
        "success": True,
        "message": "Document ingested successfully",
        "document_id": document_id,
        "filename": file.filename,
        "user_id": current_user.id,
        "ingestion": result,
    }

@router.get("/documents", response_model=List[DocumentResponse], summary="List user documents")
async def list_documents(
    current_user: UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all documents uploaded by the authenticated user."""
    rows = db.scalars(select(Document).where(Document.user_id == current_user.id)).all()
    return [
        DocumentResponse(
            id=row.id,
            user_id=row.user_id,
            file_name=row.file_name,
            created_at=row.created_at,
        )
        for row in rows
    ]

@router.get("/documents/download/{document_id}", summary="Download a document")
async def get_document(
    document_id: str,
    current_user: UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Download the original file for a document the user owns."""
    if not document_id.strip():
        raise HTTPException(status_code=400, detail="document_id is required")

    row = db.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == current_user.id,
        )
    )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    stored = _resolve_stored_file(row.file_path)
    if stored is None or not stored.is_file():
        raise HTTPException(status_code=404, detail="Document file not found")

    return FileResponse(path=stored, filename=row.file_name)

@router.delete("/documents/{document_id}", summary="Delete a document")
async def delete_document(
    document_id: str,
    current_user: UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete a document and all its vector and BM25 chunks.
    Only the owning user can delete their own documents.
    """
    if not document_id.strip():
        raise HTTPException(status_code=400, detail="document_id is required")

    components = build_components()
    vector_store = components.vector_store

    chunks_count = vector_store.count_document_chunks(document_id,metadata_filter={"user_id": current_user.id})

    if chunks_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Document not found or you do not have permission to delete it",
        )

    vector_store.delete_document(document_id)
    BM25Store(db_path="data/bm25.db").delete_document(
        document_id, user_id=current_user.id
    )

    row = db.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == current_user.id,
        )
    )
    _delete_stored_file(row.file_path if row else None)
    if row is not None:
        db.delete(row)

    return {
        "success": True,
        "message": "Document deleted successfully",
        "document_id": document_id,
        "deleted_chunks": chunks_count,
    }
