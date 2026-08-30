from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from app.retriever.reranker import CrossEncoderReranker
from app.config import Settings

from app.ingestion.chunker import (
    ChunkerService,
    LangChainRecursiveStrategy,
)

from app.ingestion.data_cleaning import (
    DataCleaningLibrary,
)

from app.ingestion.documnent_loader import (
    DocumentLoaderLibrary,
)

from app.ingestion.embedding import (
    EmbeddingService,
    HuggingFaceEmbeddingProvider,
)

from app.llm.claude import ClaudeService

from app.pipeline.pipeline import IngestionPipeline

from app.retriever.context import ContextBuilder
from app.retriever.service import RetrieverService

from app.vectorstore.chroma import ChromaVectorStore

from app.rag.graph import build_rag_graph


@dataclass(slots=True)
class RAGComponents:
    embedder: EmbeddingService
    vector_store: ChromaVectorStore


@lru_cache(maxsize=1)
def get_embedding_provider(
    model: str,
    device: str | None,
) -> HuggingFaceEmbeddingProvider:

    print("Loading embedding model...")

    return HuggingFaceEmbeddingProvider(
        model=model,
        device=device,
    )


@lru_cache(maxsize=1)
def build_components() -> RAGComponents:

    settings = Settings.from_environment()

    provider = get_embedding_provider(
        settings.embedding_model,
        settings.embedding_device,
    )

    vector_store = ChromaVectorStore(
        persist_directory=settings.chroma_persist_directory,
        collection_name=settings.collection_name_for(provider.dimension),
        embedding_dimension=provider.dimension,
        host=settings.chroma_host,
        port=settings.chroma_port,
    )

    return RAGComponents(
        embedder=EmbeddingService(provider),
        vector_store=vector_store,
    )


def build_ingestion_pipeline(
    source: str,
) -> IngestionPipeline:

    settings = Settings.from_environment()

    components = build_components()

    return IngestionPipeline(
        loader=DocumentLoaderLibrary(source=source),
        cleaner=DataCleaningLibrary(),
        chunker=ChunkerService(
            strategy=LangChainRecursiveStrategy(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        ),
        embedder=components.embedder,
        vector_store=components.vector_store,
    )


@lru_cache(maxsize=1)
def get_rag_graph():

    components = build_components()

    retriever = RetrieverService(
        embedder=components.embedder,
        vector_store=components.vector_store,
    )

    context_builder = ContextBuilder()
    reranker = CrossEncoderReranker()

    llm = ClaudeService()

    return build_rag_graph(
        retriever=retriever,
        reranker=reranker,
        context_builder=context_builder,
        llm=llm,
    )
