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
from app.models.schemas import (
    DocumentDetailResponse,
    DocumentResponse,
    DocumentUpdate,
    UserInDB
)
from app.services.auth.dependencies import get_current_user
from app.services.ingestion.document_loader import UploadRejected, scan_saved_file, validate_upload_bytes
from app.tasks.tasks import ingest_document_task, record_ingestion_failure
from database.models import Document
from database.sqlite import get_db
from app.core.config import Settings
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy import select
from .helper import (
    _content_hash,
    _find_duplicate,
    _parse_metadata_form,
    _parse_tags_form,
    _save_upload,
    _delete_stored_file,
    _resolve_stored_file,
    _stored_file_size,
    _document_response,
    _mark_failed,
)
settings = Settings.from_environment()
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Documents"])


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
async def ingest_document(file: UploadFile = File(...),tags: str | None = Form(default=None),metadata: str | None = Form(default=None),current_user: UserInDB = Depends(get_current_user),db: Session = Depends(get_db)):
    """
    Upload a document and ingest it into the vector store.

    The authenticated user's ID is stored as metadata on every chunk
    so that retrieval can be scoped per-user. Optional ``tags`` (JSON array
    or comma-separated) are stored on the document and every chunk.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    content = await file.read()
    try:
        validate_upload_bytes(
            file.filename,
            content,
            max_bytes=settings.max_upload_bytes,
        )
    except UploadRejected as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    tag_list = _parse_tags_form(tags)
    custom_metadata = _parse_metadata_form(metadata)
    content_hash = _content_hash(content)
    duplicate = _find_duplicate(db, user_id=current_user.id, content_hash=content_hash)
    if duplicate is not None:
        raise HTTPException(
            status_code=409,
            detail=f'Already uploaded as "{duplicate.file_name}".',
        )

    document_id = str(uuid4())
    file_path = _save_upload(file, content, user_id=current_user.id, document_id=document_id)
    try:
        scan_saved_file(
            Path(file_path),
            enabled=settings.clamav_enabled,
            command=settings.clamav_command,
        )
    except UploadRejected as exc:
        _delete_stored_file(file_path)
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    row = Document(
        id=document_id,
        user_id=current_user.id,
        file_name=file.filename,
        file_path=file_path,
        tags=json.dumps(tag_list) if tag_list else None,
        custom_metadata=json.dumps(custom_metadata) if custom_metadata else None,
        status="processing",
        content_hash=content_hash,
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        _delete_stored_file(file_path)
        raise HTTPException(
            status_code=409,
            detail="This file is already uploaded.",
        ) from None

    IS_ASYNC = settings.IS_ASYNC
    if IS_ASYNC:
        try:
            task = ingest_document_task.delay(
                source=file_path,
                document_id=document_id,
                filename=file.filename,
                user_id=current_user.id,
                tags=tag_list,
                custom_metadata=custom_metadata,
                content_hash=content_hash,
                file_size=len(content),
            )
        except Exception as exc:
            record_ingestion_failure(
                document_id=document_id,
                user_id=current_user.id,
                payload={"source": file_path, "filename": file.filename, "tags": tag_list},
                exc=exc,
                retries=0,
            )
            _mark_failed(db, row, file_path)
            raise
        return {
            "success": True,
            "message": "Upload received. Ingestion started in background.",
            "document_id": document_id,
            "filename": file.filename,
            "tags": tag_list,
            "metadata": custom_metadata,
            "task_id": task.id,
            "status": "processing",
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
            custom_metadata=custom_metadata,
            content_hash=content_hash,
            file_size=len(content),
        )

    try:
        result = await asyncio.to_thread(_ingest_sync)
    except Exception as exc:
        record_ingestion_failure(
            document_id=document_id,
            user_id=current_user.id,
            payload={"source": file_path, "filename": file.filename, "tags": tag_list},
            exc=exc,
            retries=0,
        )
        _mark_failed(db, row, file_path)
        raise

    row.status = "ready"
    row.error_message = None
    db.commit()

    return {
        "success": True,
        "message": "Document ingested successfully",
        "document_id": document_id,
        "filename": file.filename,
        "user_id": current_user.id,
        "tags": tag_list,
        "metadata": custom_metadata,
        "status": "ready",
        "ingestion": result,
    }

@router.get("/documents", response_model=List[DocumentResponse], summary="List user documents")
async def list_documents(current_user: UserInDB = Depends(get_current_user),db: Session = Depends(get_db)):
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
        _document_response(
            row,
            chunk_count=chunk_counts.get(row.id, 0),
            file_size=_stored_file_size(row.file_path),
        )
        for row in rows
    ]


def _owned_document(db: Session, document_id: str, user_id: str) -> Document:
    if not document_id.strip():
        raise HTTPException(status_code=400, detail="document_id is required")
    row = db.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == user_id,
        )
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return row


@router.get("/documents/{document_id}",response_model=DocumentDetailResponse,summary="Show one document",)
async def show_document(document_id: str,current_user: UserInDB = Depends(get_current_user),db: Session = Depends(get_db)):
    """Return status, tags, custom metadata, and a short text preview."""
    row = _owned_document(db, document_id, current_user.id)

    def _detail() -> tuple[int, str]:
        vector_store = build_components().vector_store
        owner = {"user_id": current_user.id}
        try:
            count = vector_store.count_document_chunks(row.id, metadata_filter=owner)
        except Exception:
            logger.warning("Chunk count failed for document %s", row.id, exc_info=True)
            count = 0
        preview = ""
        if row.status == "ready":
            try:
                preview = vector_store.preview_document_text(row.id, current_user.id)
            except Exception:
                logger.warning("Preview failed for document %s", row.id, exc_info=True)
        return count, preview

    chunk_count, preview = await asyncio.to_thread(_detail)
    base = _document_response(
        row,
        chunk_count=chunk_count,
        file_size=_stored_file_size(row.file_path),
    )
    return DocumentDetailResponse(
        **base.model_dump(),
        preview=preview or None,
    )


@router.patch("/documents/{document_id}",response_model=DocumentResponse,summary="Update document tags and custom metadata")
async def update_document(document_id: str,body: DocumentUpdate,current_user: UserInDB = Depends(get_current_user),db: Session = Depends(get_db)):
    """Save tag chips and custom metadata chips for one owned document."""
    row = _owned_document(db, document_id, current_user.id)
    if row.status == "processing":
        raise HTTPException(status_code=409, detail="Document is still processing.")

    if body.tags is not None and row.status == "ready":
        tags = body.tags

        def _sync_tags() -> None:
            build_components().vector_store.set_document_tags(
                row.id, current_user.id, tags
            )

        try:
            await asyncio.to_thread(_sync_tags)
        except Exception:
            logger.warning("Could not update chunk tags for %s", row.id, exc_info=True)
            raise HTTPException(status_code=502, detail="Could not update document tags.") from None
        row.tags = json.dumps(tags) if tags else None

    if body.metadata is not None:
        row.custom_metadata = json.dumps(body.metadata) if body.metadata else None

    db.commit()
    db.refresh(row)

    def _count() -> int:
        if row.status != "ready":
            return 0
        try:
            return build_components().vector_store.count_document_chunks(
                row.id, metadata_filter={"user_id": current_user.id}
            )
        except Exception:
            logger.warning("Chunk count failed for document %s", row.id, exc_info=True)
            return 0

    chunk_count = await asyncio.to_thread(_count)
    return _document_response(
        row,
        chunk_count=chunk_count,
        file_size=_stored_file_size(row.file_path),
    )

@router.get("/documents/download/{document_id}", summary="Download a document")
async def get_document(document_id: str,current_user: UserInDB = Depends(get_current_user),db: Session = Depends(get_db)):
    """Download the original file for a document the user owns."""
    if not document_id.strip():
        raise HTTPException(status_code=400, detail="document_id is required")

    row = db.scalar(select(Document).where(Document.id == document_id,Document.user_id == current_user.id))
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    stored = _resolve_stored_file(row.file_path)
    if stored is None or not stored.is_file():
        raise HTTPException(status_code=404, detail="Document file not found")

    return FileResponse(path=stored, filename=row.file_name)

@router.delete("/documents/{document_id}", summary="Delete a document")
async def delete_document(document_id: str,current_user: UserInDB = Depends(get_current_user),db: Session = Depends(get_db)):
    """
    Delete a document: its Qdrant points (dense + sparse vectors live on the
    same point), the stored upload, and the SQLite row.
    Only the owning user can delete their own documents.
    """
    if not document_id.strip():
        raise HTTPException(status_code=400, detail="document_id is required")

    row = db.scalar(select(Document).where(Document.id == document_id,Document.user_id == current_user.id))
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
