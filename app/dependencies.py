"""Composition root: construct the same RAG components in every process."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from app.config import Settings
from app.ingestion.chunker import ChunkerService, LangChainRecursiveStrategy
from app.ingestion.data_cleaning import DataCleaningLibrary
from app.ingestion.documnent_loader import DocumentLoaderLibrary
from app.ingestion.embedding import EmbeddingService, HuggingFaceEmbeddingProvider
from app.llm.claude import ClaudeService
from app.pipeline.pipeline import IngestionPipeline
from app.query.service import QueryService
from app.retriever.context import ContextBuilder
from app.retriever.service import RetrieverService
from app.vectorstore.chroma import ChromaVectorStore

@dataclass(slots=True)
class RAGComponents:
    embedder: EmbeddingService
    vector_store: ChromaVectorStore

@lru_cache(maxsize=4)
def get_embedding_provider(
    model: str,
    device: str | None,
) -> HuggingFaceEmbeddingProvider:
    """Load one embedding model per process, reused by queries and tasks."""
    return HuggingFaceEmbeddingProvider(model=model, device=device)


def build_components(settings: Settings | None = None) -> RAGComponents:
    settings = settings or Settings.from_environment()
    provider = get_embedding_provider(settings.embedding_model, settings.embedding_device)
    vector_store = ChromaVectorStore(
        persist_directory=settings.chroma_persist_directory,
        collection_name=settings.collection_name_for(provider.dimension),
        embedding_dimension=provider.dimension,
        host=settings.chroma_host,
        port=settings.chroma_port,
    )
    return RAGComponents(embedder=EmbeddingService(provider), vector_store=vector_store)

def build_ingestion_pipeline(source: str) -> IngestionPipeline:
    settings = Settings.from_environment()
    components = build_components(settings)
    return IngestionPipeline(
        loader=DocumentLoaderLibrary(source=source), cleaner=DataCleaningLibrary(),
        chunker=ChunkerService(strategy=LangChainRecursiveStrategy(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)),
        embedder=components.embedder, vector_store=components.vector_store,
    )

@lru_cache(maxsize=1)
def get_query_service() -> QueryService:
    """Create the API query service once per web process."""
    components = build_components()
    return QueryService(
        retriever=RetrieverService(embedder=components.embedder, vector_store=components.vector_store),
        context_builder=ContextBuilder(), llm=ClaudeService(),
    )
