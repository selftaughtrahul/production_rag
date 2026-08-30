from app.tasks.celery_app import celery_app

@celery_app.task(name="app.tasks.ingest_document_task")
def ingest_document_task(file_path: str) -> dict:
    """Celery background task to ingest a document."""
    from app.api.dependencies import build_ingestion_pipeline
    return build_ingestion_pipeline(source=file_path).ingest(source=file_path)
