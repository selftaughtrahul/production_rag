import json
import logging
from pathlib import Path

from app.tasks.celery_app import _ensure_project_root, celery_app

logger = logging.getLogger(__name__)
_DOCUMENTS_ROOT = Path("documents")


def _mark_document(document_id: str, **fields: object) -> None:
    """Update an ingestion row created before the background task starts."""
    from database.models import Document
    from database.sqlite import SessionLocal

    session = SessionLocal()
    try:
        row = session.get(Document, document_id)
        if row is None:
            return
        for key, value in fields.items():
            setattr(row, key, value)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _delete_stored_file(file_path: str | None) -> None:
    if not file_path:
        return
    try:
        resolved = Path(file_path).resolve()
        root = _DOCUMENTS_ROOT.resolve()
        if root not in resolved.parents:
            return
        resolved.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove stored upload %s", file_path, exc_info=True)


@celery_app.task(name="app.tasks.ingest_document_task")
def ingest_document_task(
    source: str,
    document_id: str,
    filename: str,
    user_id: str,
    tags: list[str] | None = None,
) -> dict:
    """Celery background task to ingest a document with tenant metadata and tags."""
    _ensure_project_root()
    from app.api.dependencies import build_ingestion_pipeline
    from database.models import Document
    from database.sqlite import SessionLocal

    tags = list(tags or [])
    try:
        result = build_ingestion_pipeline(source=source).ingest(
            source=source,
            document_id=document_id,
            filename=filename,
            user_id=user_id,
            tags=tags,
        )
    except Exception:
        _mark_document(
            document_id,
            status="failed",
            error_message="Ingestion failed.",
            content_hash=None,
        )
        _delete_stored_file(source)
        raise

    session = SessionLocal()
    try:
        row = session.get(Document, document_id)
        if row is None:
            session.add(
                Document(
                    id=document_id,
                    user_id=user_id,
                    file_name=filename,
                    file_path=source,
                    tags=json.dumps(tags) if tags else None,
                    status="ready",
                )
            )
        else:
            row.status = "ready"
            row.error_message = None
            row.file_path = source
            row.tags = json.dumps(tags) if tags else None
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

    return result
