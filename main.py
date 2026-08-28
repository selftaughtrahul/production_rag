from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, HTTPException, UploadFile
from app.vectorstore.chroma import ChromaVectorStore
from app.pipeline.pipeline import IngestionPipeline
from app.ingestion.documnent_loader import DocumentLoaderLibrary, DocumentLoaderCustom
from app.ingestion.data_cleaning import DataCleaningLibrary, DataCleaningCustom
from app.ingestion.chunker import (
    RecursiveCharacterTextSplitter,
    ChunkerService,
    FixedSizeChunkingStrategy,
    RecursiveChunkingStrategy,
    LangChainRecursiveStrategy,
)
from app.ingestion.embedding import (
    OpenAIEmbeddingProvider,
    HuggingFaceEmbeddingProvider,
    EmbeddingService,
)


app = FastAPI(
    title="RAG API",
    description="Document ingestion API",
    version="1.0.0",
)

cleaner = DataCleaningLibrary()
chunker = ChunkerService(strategy=LangChainRecursiveStrategy())
embedder = EmbeddingService(provider=HuggingFaceEmbeddingProvider())


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
    Upload a document and ingest it into the vector database.
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

    # --------------------------------------------------
    # 2. Save temporary file
    # --------------------------------------------------
    temp_file_path = save_file(file, content)

    # --------------------------------------------------
    # 3. Create vector store
    # --------------------------------------------------
    vector_store = ChromaVectorStore(
        persist_directory="data/chroma_db",
        collection_name="rag",
        embedding_dimension=384,
    )

    # --------------------------------------------------
    # 4. Create pipeline dependencies
    # --------------------------------------------------
    loader = DocumentLoaderLibrary(source=temp_file_path)

    # --------------------------------------------------
    # 5. Create ingestion pipeline
    # --------------------------------------------------

    pipeline = IngestionPipeline(
        loader=loader,
        cleaner=cleaner,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    # --------------------------------------------------
    # 6. Run ingestion
    # --------------------------------------------------

    result = await pipeline.ingest(
        source=temp_file_path,
    )

    return {
        "success": True,
        "message": "Document ingested successfully",
        "filename": file.filename,
        "result": result,
    }

    # except HTTPException:
    #     raise

    # except Exception as exc:
    #     raise HTTPException(
    #         status_code=500,
    #         detail=f"Document ingestion failed: {exc}",
    #     )
