from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, HTTPException, UploadFile
from app.tasks.tasks import ingest_document_task
from celery.result import AsyncResult
from app.api.query import router as query_router
from app.api.dependencies import build_ingestion_pipeline
from pathlib import Path
from uuid import uuid4


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


@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
):
    """
    Upload a document and start ingestion.
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

    # ---------------------------------------------------------
    # 1. Generate ID for the complete document
    # ---------------------------------------------------------

    document_id = str(uuid4())

    # Keep the original filename
    original_filename = file.filename

    # ---------------------------------------------------------
    # 2. Save temporary file
    # ---------------------------------------------------------

    temp_file_path = save_file(
        file,
        content,
    )

    # ---------------------------------------------------------
    # 3. Start ingestion
    # ---------------------------------------------------------

    is_linux = False

    if is_linux:

        task = ingest_document_task.delay(
            source=temp_file_path,
            document_id=document_id,
            filename=original_filename,
        )

        return {
            "success": True,
            "message": (
                "Document upload successful. " "Ingestion started in the background."
            ),
            "document_id": document_id,
            "filename": original_filename,
            "task_id": task.id,
            "status": task.status,
        }

    # ---------------------------------------------------------
    # Development / synchronous mode
    # ---------------------------------------------------------

    pipeline = build_ingestion_pipeline(
        source=temp_file_path,
    )

    result = pipeline.ingest(
        source=temp_file_path,
        document_id=document_id,
        filename=original_filename,
    )

    return {
        "success": True,
        "message": "Document upload successful",
        "document_id": document_id,
        "filename": original_filename,
        "ingestion": result,
    }


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
):
    """
    Delete a complete document and all of its vectors.
    """

    if not document_id.strip():
        raise HTTPException(
            status_code=400,
            detail="document_id is required",
        )

    from app.api.dependencies import build_components

    components = build_components()

    vector_store = components.vector_store

    # Check whether the document exists
    chunks_count = vector_store.count_document_chunks(document_id)

    if chunks_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    # Delete ALL chunks belonging to this document
    vector_store.delete_document(document_id)

    return {
        "success": True,
        "message": "Document deleted successfully",
        "document_id": document_id,
        "deleted_chunks": chunks_count,
    }
