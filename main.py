from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, HTTPException, UploadFile
from app.tasks.tasks import ingest_document_task
from celery.result import AsyncResult
from app.api.query import router as query_router


app = FastAPI(
    title="RAG API",
    description="Document ingestion API",
    version="1.0.0",
)

app.include_router(query_router)


def save_file(file, content):
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


@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
):
    """
    Upload a document and start background ingestion.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty",
        )

    # 1. Save temporary file
    temp_file_path = save_file(file, content)

    # 2. Trigger background ingestion task
    is_linux = False
    if is_linux:
        task = ingest_document_task.delay(temp_file_path)
        return {
            "success": True,
            "message": "Document upload successful. Ingestion started in the background.",
            "filename": file.filename,
            "task_id": task.id,
            "status": task.status,
        }

    else:
        from app.api.dependencies import build_ingestion_pipeline

        build_ingestion_pipeline(source=temp_file_path).ingest(source=temp_file_path)
        return {
            "success": True,
            "message": "Document upload successful",
            "filename": file.filename,
        }


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the status and result of a background ingestion task.
    """
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
