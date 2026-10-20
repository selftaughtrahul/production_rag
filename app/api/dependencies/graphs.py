import asyncio
from functools import lru_cache
from typing import Any, Optional

from fastapi import Depends
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agent.multi_agent.master_graph import build_master_agent_graph
from app.core.config import Settings
from app.core.device import resolve_torch_device
from app.guardrails.factory import build_input_guardrails
from app.services.llm.claude import ClaudeService
from app.services.rag.graph import build_rag_graph
from app.services.retriever.base import Retriever
from app.services.retriever.context import ContextBuilder
from app.services.retriever.qdrant_retreiver import QdrantHybridRetriever
from app.services.retriever.reranker import CrossEncoderReranker
from app.tools.registry import ToolRegistry, build_default_tool_registry

from .checkpointer import get_checkpointer
from .components import build_components, get_cross_encoder_model
from .guardrails import build_wired_output_guardrails, get_nemo_provider, get_presidio_provider
from .llm import get_laya, get_llm

_compiled_graphs: dict[str, Any] = {}
_graph_lock = asyncio.Lock()
_tool_registry_instance: Optional[ToolRegistry] = None


def _build_common_rag_graph(retriever: Retriever, checkpointer: AsyncSqliteSaver) -> Any:
    settings = Settings.from_environment()
    reranker_model = get_cross_encoder_model(
        model=settings.reranker_model,
        device=resolve_torch_device(settings.embedding_device),
    )
    return build_rag_graph(
        retriever=retriever,
        reranker=CrossEncoderReranker(model=reranker_model),
        context_builder=ContextBuilder(),
        llm=ClaudeService(),
        input_guardrails=build_input_guardrails(nemo_provider=get_nemo_provider()),
        output_guardrails=build_wired_output_guardrails(),
        checkpointer=checkpointer,
    )


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> Retriever:
    components = build_components()
    return QdrantHybridRetriever(
        embedder=components.embedder,
        vector_store=components.vector_store,
    )


async def get_hybrid_rag_graph() -> Any:
    """Hybrid retrieval graph: Dense + BM25 → RRF Fusion → Reranker → LLM."""
    if "hybrid" not in _compiled_graphs:
        async with _graph_lock:
            if "hybrid" not in _compiled_graphs:
                _compiled_graphs["hybrid"] = _build_common_rag_graph(
                    get_hybrid_retriever(),
                    await get_checkpointer(),
                )
    return _compiled_graphs["hybrid"]


def get_tool_registry(
    retriever: Retriever = Depends(get_hybrid_retriever),
) -> ToolRegistry:
    """Dependency provider for the ToolRegistry instance."""
    global _tool_registry_instance
    if _tool_registry_instance is None:
        _tool_registry_instance = build_default_tool_registry(retriever=retriever)
    return _tool_registry_instance


async def get_master_agent_graph() -> Any:
    """Singleton compiled master multi-agent graph."""
    if "agent" in _compiled_graphs:
        return _compiled_graphs["agent"]

    compiled_rag = await get_hybrid_rag_graph()
    checkpointer = await get_checkpointer()
    async with _graph_lock:
        if "agent" not in _compiled_graphs:
            retriever = get_hybrid_retriever()
            _compiled_graphs["agent"] = build_master_agent_graph(
                llm=get_llm(),
                router=get_laya(),
                tool_registry=get_tool_registry(retriever=retriever),
                compiled_rag_graph=compiled_rag,
                input_guardrails=build_input_guardrails(
                    nemo_provider=get_nemo_provider(),
                    presidio_provider=get_presidio_provider(),
                ),
                output_guardrails=build_wired_output_guardrails(),
                checkpointer=checkpointer,
            )
    return _compiled_graphs["agent"]
