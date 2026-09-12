from app.tasks.celery_app import celery_app


@celery_app.task(name="app.tasks.ingest_document_task")
def ingest_document_task(source: str, document_id: str, filename: str, user_id: str) -> dict:
    """Celery background task to ingest a document with tenant metadata."""
    from app.api.dependencies import build_ingestion_pipeline
    from database.models import Document
    from database.sqlite import SessionLocal

    result = build_ingestion_pipeline(source=source).ingest(
        source=source,
        document_id=document_id,
        filename=filename,
        user_id=user_id,
    )

    session = SessionLocal()
    try:
        session.add(
            Document(
                id=document_id,
                user_id=user_id,
                file_name=filename,
                file_path=source,
            )
        )
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return result
