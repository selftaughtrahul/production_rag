import json
import logging
import os
import shutil
from pathlib import Path
from uuid import uuid4

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
        images = resolved.parent / f"{resolved.stem}_images"
        if images.is_dir() and root in images.resolve().parents:
            shutil.rmtree(images, ignore_errors=True)
    except OSError:
        logger.warning("Could not remove stored upload %s", file_path, exc_info=True)


def is_retryable_ingest_error(exc: BaseException) -> bool:
    """Transient IO and broker errors can be retried. Bad files cannot."""
    if isinstance(exc, (FileNotFoundError, PermissionError, ValueError, RuntimeError)):
        return False
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


def _max_retries() -> int:
    try:
        return max(0, int(os.getenv("INGEST_MAX_RETRIES", "3")))
    except ValueError:
        return 3


def record_ingestion_failure(
    *,
    document_id: str,
    user_id: str,
    payload: dict[str, object],
    exc: BaseException,
    retries: int,
) -> None:
    """Persist a terminal ingest failure. Recording must not hide the original error."""
    from database.models import IngestionDeadLetter
    from database.sqlite import SessionLocal

    message = (str(exc).strip() or type(exc).__name__)[:500]
    session = SessionLocal()
    try:
        session.add(
            IngestionDeadLetter(
                id=str(uuid4()),
                document_id=document_id,
                user_id=user_id,
                payload=json.dumps(payload),
                error_type=type(exc).__name__,
                error_message=message,
                retries=retries,
            )
        )
        session.commit()
    except Exception:
        session.rollback()
        logger.warning("Could not record dead letter for %s", document_id, exc_info=True)
    finally:
        session.close()


@celery_app.task(bind=True, name="app.tasks.ingest_document_task", max_retries=_max_retries())
def ingest_document_task(
    self,
    source: str,
    document_id: str,
    filename: str,
    user_id: str,
    content_hash: str | None = None,
    file_size: int | None = None,
) -> dict:
    """Celery background task to ingest a document and store prepared metadata."""
    _ensure_project_root()
    from app.api.dependencies import build_ingestion_pipeline
    from database.models import Document
    from database.sqlite import SessionLocal

    try:
        result = build_ingestion_pipeline(source=source).ingest(
            source=source,
            document_id=document_id,
            filename=filename,
            user_id=user_id,
            content_hash=content_hash,
            file_size=file_size,
        )
    except Exception as exc:
        if is_retryable_ingest_error(exc) and self.request.retries < self.max_retries:
            logger.warning(
                "Retrying ingest for %s (attempt %s)",
                document_id,
                self.request.retries + 1,
            )
            raise self.retry(exc=exc, countdown=min(60, 2 ** self.request.retries))
        record_ingestion_failure(
            document_id=document_id,
            user_id=user_id,
            payload={
                "source": source,
                "filename": filename,
            },
            exc=exc,
            retries=self.request.retries,
        )
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
        prepared = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        stored_metadata = json.dumps(prepared) if prepared else None
        row = session.get(Document, document_id)
        if row is None:
            session.add(
                Document(
                    id=document_id,
                    user_id=user_id,
                    file_name=filename,
                    file_path=source,
                    custom_metadata=stored_metadata,
                    status="ready",
                )
            )
        else:
            row.status = "ready"
            row.error_message = None
            row.file_path = source
            row.custom_metadata = stored_metadata
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

    return result
