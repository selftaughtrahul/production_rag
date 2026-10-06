"""
Document management routes.

 
"""

import asyncio
import json
import logging
from pathlib import Path
from uuid import uuid4
from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from typing import List
from app.api.dependencies import build_components, build_ingestion_pipeline
from app.models.schemas import DocumentResponse, UserInDB, normalize_tags
from app.services.auth.dependencies import get_current_user
from app.tasks.tasks import ingest_document_task
from database.models import Document
from database.sqlite import get_db
from app.core.config import Settings
from sqlalchemy.orm import Session
from sqlalchemy import select
settings = Settings.from_environment()
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Documents"])

_DOCUMENTS_ROOT = Path("documents")


def _parse_tags_form(raw: str | None) -> list[str]:
    """Accept a JSON array or a comma-separated string from the upload form."""
    if not raw or not raw.strip():
        return []
    text = raw.strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="tags must be a JSON array") from exc
        if not isinstance(parsed, list):
            raise HTTPException(status_code=400, detail="tags must be a JSON array")
        return normalize_tags(parsed)
    return normalize_tags(text.split(","))


def _row_tags(row: Document) -> list[str]:
    """Decode the JSON tags column; tolerate legacy rows without tags."""
    if not row.tags:
        return []
    try:
        parsed = json.loads(row.tags)
    except json.JSONDecodeError:
        return []
    return normalize_tags(parsed) if isinstance(parsed, list) else []


def _stored_file_size(file_path: str | None) -> int | None:
    stored = _resolve_stored_file(file_path)
    if stored is None or not stored.is_file():
        return None
    try:
        return stored.stat().st_size
    except OSError:
        return None


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
        logger.warning("Could not remove stored upload %s", resolved, exc_info=True)



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
    tags: str | None = Form(default=None),
    current_user: UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Upload a document and ingest it into the vector store.

    The authenticated user's ID is stored as metadata on every chunk
    so that retrieval can be scoped per-user. Optional ``tags`` (JSON array
    or comma-separated) are stored on the document and every chunk.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    tag_list = _parse_tags_form(tags)
    document_id = str(uuid4())
    file_path = _save_upload(file,content,user_id=current_user.id,document_id=document_id)

    IS_ASYNC = settings.IS_ASYNC
    if IS_ASYNC:
        task = ingest_document_task.delay(
            source=file_path,
            document_id=document_id,
            filename=file.filename,
            user_id=current_user.id,
            tags=tag_list,
        )
        return {
            "success": True,
            "message": "Upload received. Ingestion started in background.",
            "document_id": document_id,
            "filename": file.filename,
            "tags": tag_list,
            "task_id": task.id,
            "status": task.status,
        }

    # ── Synchronous mode (dev / Windows) ───────────────────
    def _ingest_sync() -> dict:
        pipeline = build_ingestion_pipeline(source=file_path)
        return pipeline.ingest(
            source=file_path,
            document_id=document_id,
            filename=file.filename,
            user_id=current_user.id,
            tags=tag_list,
        )

    try:
        result = await asyncio.to_thread(_ingest_sync)
    except Exception:
        _delete_stored_file(file_path)
        raise

    db.add(
        Document(
            id=document_id,
            user_id=current_user.id,
            file_name=file.filename,
            file_path=file_path,
            tags=json.dumps(tag_list) if tag_list else None,
        )
    )

    return {
        "success": True,
        "message": "Document ingested successfully",
        "document_id": document_id,
        "filename": file.filename,
        "user_id": current_user.id,
        "tags": tag_list,
        "ingestion": result,
    }

@router.get("/documents", response_model=List[DocumentResponse], summary="List user documents")
async def list_documents(
    current_user: UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List the user's documents with tags, file size, and stored chunk count."""
    rows = db.scalars(
        select(Document)
        .where(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
    ).all()

    def _chunk_counts() -> dict[str, int]:
        if not rows:
            return {}
        vector_store = build_components().vector_store
        counts: dict[str, int] = {}
        for row in rows:
            try:
                counts[row.id] = vector_store.count_document_chunks(
                    row.id, metadata_filter={"user_id": current_user.id}
                )
            except Exception:
                logger.warning("Chunk count failed for document %s", row.id, exc_info=True)
                counts[row.id] = 0
        return counts

    chunk_counts = await asyncio.to_thread(_chunk_counts)

    return [
        DocumentResponse(
            id=row.id,
            user_id=row.user_id,
            file_name=row.file_name,
            created_at=row.created_at,
            tags=_row_tags(row),
            file_size=_stored_file_size(row.file_path),
            chunk_count=chunk_counts.get(row.id, 0),
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
    Delete a document: its Qdrant points (dense + sparse vectors live on the
    same point), the stored upload, and the SQLite row.
    Only the owning user can delete their own documents.
    """
    if not document_id.strip():
        raise HTTPException(status_code=400, detail="document_id is required")

    row = db.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == current_user.id,
        )
    )
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found or you do not have permission to delete it",
        )

    def _delete_vectors() -> int:
        vector_store = build_components().vector_store
        return vector_store.delete_document(document_id, user_id=current_user.id)

    deleted_chunks = await asyncio.to_thread(_delete_vectors)

    _delete_stored_file(row.file_path)
    db.delete(row)

    return {
        "success": True,
        "message": "Document deleted successfully",
        "document_id": document_id,
        "deleted_chunks": deleted_chunks,
    }
