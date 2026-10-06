import json

from app.tasks.celery_app import _ensure_project_root, celery_app


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
    result = build_ingestion_pipeline(source=source).ingest(
        source=source,
        document_id=document_id,
        filename=filename,
        user_id=user_id,
        tags=tags,
    )

    session = SessionLocal()
    try:
        session.add(
            Document(
                id=document_id,
                user_id=user_id,
                file_name=filename,
                file_path=source,
                tags=json.dumps(tags) if tags else None,
            )
        )
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

    return result
