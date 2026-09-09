import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import aiosqlite
from fastapi import Depends
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sentence_transformers import CrossEncoder

from app.core.config import Settings
from app.core.device import resolve_torch_device
from app.guardrails.factory import (
    build_input_guardrails,
    build_output_guardrails,
)
from app.services.ingestion.chunker import ChunkerService, LangChainRecursiveStrategy
from app.services.ingestion.data_cleaning import DataCleaningLibrary
from app.services.ingestion.document_loader import DocumentLoaderLibrary
from app.services.ingestion.embedding import EmbeddingService, HuggingFaceEmbeddingProvider
from app.services.ingestion.pipeline import IngestionPipeline
from app.services.llm.claude import ClaudeService
from app.services.rag.graph import build_rag_graph
from app.services.retriever.bm25 import BM25Retriever
from app.services.retriever.bm25_store import BM25Store
from app.services.retriever.context import ContextBuilder
from app.services.retriever.dense import DenseRetriever
from app.services.retriever.hybrid import HybridRetriever
from app.services.retriever.reranker import CrossEncoderReranker
from app.services.vectorstore.chroma import ChromaVectorStore
from app.tools.registry import ToolRegistry, build_default_tool_registry
from database.sqlite import DB_FILE_PATH
from database.sqlite import engine as sqlite_engine

from app.agent.multi_agent.master_graph import build_master_agent_graph

_checkpointer: AsyncSqliteSaver | None = None
_checkpointer_lock = asyncio.Lock()
_compiled_graphs: dict[str, Any] = {}
_graph_lock = asyncio.Lock()


async def get_checkpointer() -> AsyncSqliteSaver:
    """Lazy async SQLite checkpointer. SqliteSaver cannot be used with ainvoke."""
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    async with _checkpointer_lock:
        if _checkpointer is None:
            conn = await aiosqlite.connect(DB_FILE_PATH)
            saver = AsyncSqliteSaver(conn)
            await saver.setup()
            _checkpointer = saver
    return _checkpointer


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
    resolved = resolve_torch_device(device)
    print(f"Loading cross encoder model: {model} on {resolved}...")
    return CrossEncoder(model, device=resolved)


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


def _build_common_rag_graph(retriever, checkpointer: AsyncSqliteSaver):
    """Helper to build a RAG graph with shared components."""
    settings = Settings.from_environment()
    context_builder = ContextBuilder()

    reranker_model = get_cross_encoder_model(
        model=settings.reranker_model,
        device=resolve_torch_device(settings.embedding_device),
    )
    reranker = CrossEncoderReranker(model=reranker_model)

    llm = ClaudeService()
    input_guardrail = build_input_guardrails()
    output_guardrail = build_output_guardrails()

    return build_rag_graph(
        retriever=retriever,
        reranker=reranker,
        context_builder=context_builder,
        llm=llm,
        input_guardrails=input_guardrail,
        output_guardrails=output_guardrail,
        checkpointer=checkpointer,
    )


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    """Singleton getter for the Hybrid Retriever (Dense + BM25)."""
    components = build_components()

    dense_retriever = DenseRetriever(
        embedder=components.embedder,
        vector_store=components.vector_store,
    )

    bm25_store = BM25Store(db_path="data/bm25.db")
    bm25_retriever = BM25Retriever(bm25_store=bm25_store)

    return HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )


async def get_rag_graph():
    """Single dense-vector retrieval graph."""
    if "rag" not in _compiled_graphs:
        async with _graph_lock:
            if "rag" not in _compiled_graphs:
                components = build_components()
                retriever = DenseRetriever(
                    embedder=components.embedder,
                    vector_store=components.vector_store,
                )
                checkpointer = await get_checkpointer()
                _compiled_graphs["rag"] = _build_common_rag_graph(retriever, checkpointer)
    return _compiled_graphs["rag"]


async def get_hybrid_rag_graph():
    """Hybrid retrieval graph: Dense + BM25 → RRF Fusion → Reranker → LLM."""
    if "hybrid" not in _compiled_graphs:
        async with _graph_lock:
            if "hybrid" not in _compiled_graphs:
                retriever = get_hybrid_retriever()
                checkpointer = await get_checkpointer()
                _compiled_graphs["hybrid"] = _build_common_rag_graph(
                    retriever, checkpointer
                )
    return _compiled_graphs["hybrid"]


@lru_cache(maxsize=1)
def get_llm() -> ClaudeService:
    """Return shared ClaudeService instance."""
    return ClaudeService()


_tool_registry_instance: Optional[ToolRegistry] = None


def get_tool_registry(
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
) -> ToolRegistry:
    """Dependency provider for the ToolRegistry instance."""
    global _tool_registry_instance
    if _tool_registry_instance is None:
        _tool_registry_instance = build_default_tool_registry(
            retriever=retriever, db_engine=sqlite_engine
        )
    return _tool_registry_instance


async def get_master_agent_graph():
    """
    Singleton provider for the compiled Master Multi-Agent Graph.
    Equipped with:
      - Supervisor-level input & output guardrails
      - Supervisor-level conversational memory checkpointer
      - Long-term memory extraction & SQLite persistence
      - Specialized sub-agents (RAG, SQL, Web, General)
    """
    if "agent" in _compiled_graphs:
        return _compiled_graphs["agent"]

    compiled_rag = await get_hybrid_rag_graph()
    checkpointer = await get_checkpointer()
    async with _graph_lock:
        if "agent" not in _compiled_graphs:
            llm = get_llm()
            retriever = get_hybrid_retriever()
            tool_registry = get_tool_registry(retriever=retriever)
            _compiled_graphs["agent"] = build_master_agent_graph(
                llm=llm,
                tool_registry=tool_registry,
                compiled_rag_graph=compiled_rag,
                input_guardrails=build_input_guardrails(),
                output_guardrails=build_output_guardrails(),
                checkpointer=checkpointer,
            )
    return _compiled_graphs["agent"]



