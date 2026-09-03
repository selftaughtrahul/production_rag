from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver
from sentence_transformers import CrossEncoder

from app.core.config import Settings
from app.services.ingestion.chunker import ChunkerService, LangChainRecursiveStrategy
from app.services.ingestion.data_cleaning import DataCleaningLibrary
from app.services.ingestion.document_loader import DocumentLoaderLibrary
from app.services.ingestion.embedding import EmbeddingService, HuggingFaceEmbeddingProvider
from app.services.ingestion.pipeline import IngestionPipeline

from app.services.retriever.bm25 import BM25Retriever
from app.services.retriever.bm25_store import BM25Store
from app.services.retriever.context import ContextBuilder
from app.services.retriever.dense import DenseRetriever
from app.services.retriever.hybrid import HybridRetriever
from app.services.retriever.reranker import CrossEncoderReranker

from app.services.llm.claude import ClaudeService
from app.services.rag.graph import build_rag_graph
from app.services.vectorstore.chroma import ChromaVectorStore


from database.sqlite import get_connection

# Initialize SQLite checkpointer for conversational memory globally
_db_conn = get_connection()
checkpointer = SqliteSaver(_db_conn)


@dataclass(slots=True)
class RAGComponents:
    embedder: EmbeddingService
    vector_store: ChromaVectorStore


@lru_cache(maxsize=1)
def get_embedding_provider(model: str, device: str | None) -> HuggingFaceEmbeddingProvider:
    print("Loading embedding model...")
    return HuggingFaceEmbeddingProvider(
        model=model,
        device=device,
    )


@lru_cache(maxsize=1)
def get_cross_encoder_model(model: str, device: str | None = None) -> CrossEncoder:
    print(f"Loading cross encoder model: {model}...")
    return CrossEncoder(model, device=device)


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


def build_ingestion_pipeline(source: str) -> IngestionPipeline:
    settings = Settings.from_environment()
    components = build_components()
    bm25_store = BM25Store(db_path="data/bm25.db")
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
        bm25_store=bm25_store,
    )


def _build_common_rag_graph(retriever):
    """Helper to build a RAG graph with shared components."""
    settings = Settings.from_environment()
    context_builder = ContextBuilder()
    
    reranker_model = get_cross_encoder_model(
        model=settings.reranker_model,
        device=settings.embedding_device,
    )
    reranker = CrossEncoderReranker(model=reranker_model)
    
    llm = ClaudeService()

    return build_rag_graph(
        retriever=retriever,
        reranker=reranker,
        context_builder=context_builder,
        llm=llm,
        checkpointer=checkpointer,
    )


@lru_cache(maxsize=1)
def get_rag_graph():
    """Single dense-vector retrieval graph (original)."""
    components = build_components()

    retriever = DenseRetriever(
        embedder=components.embedder,
        vector_store=components.vector_store,
    )

    return _build_common_rag_graph(retriever)


@lru_cache(maxsize=1)
def get_hybrid_rag_graph():
    """
    Hybrid retrieval graph: Dense + BM25 → RRF Fusion → Reranker → LLM.

    Use this via Depends(get_hybrid_rag_graph) on any endpoint.
    """
    components = build_components()

    # Dense retriever
    dense_retriever = DenseRetriever(
        embedder=components.embedder,
        vector_store=components.vector_store,
    )

    # BM25 retriever
    bm25_store = BM25Store(db_path="data/bm25.db")
    bm25_retriever = BM25Retriever(bm25_store=bm25_store)

    # Hybrid = Dense + BM25 fused with RRF
    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )

    return _build_common_rag_graph(retriever)
