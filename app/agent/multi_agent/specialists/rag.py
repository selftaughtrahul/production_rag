"""Internal-document specialist."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage

from app.agent.multi_agent.state import AgentOutput, SupervisorState
from app.tools.execute import invoke_tool
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_LIST_HINTS = ("list my documents", "list documents", "my uploads", "uploaded files")


class RAGSubGraphNode:
    """Invoke the nested CRAG graph or a narrow document metadata tool."""

    def __init__(
        self,
        compiled_rag_graph: Any | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.rag_graph = compiled_rag_graph
        self.registry = tool_registry

    async def __call__(self, state: SupervisorState) -> dict[str, Any]:
        query = state.get("query", "")
        user_id = state.get("user_id", "")
        thread_id = state.get("thread_id", "")

        if any(hint in query.lower() for hint in _LIST_HINTS):
            tool = (
                self.registry.get_tool_for_agent("rag_agent", "list_user_documents")
                if self.registry
                else None
            )
            answer = (
                await invoke_tool(
                    tool,
                    {"user_id": user_id, "limit": 50},
                    user_id=user_id,
                    session_id=thread_id,
                )
                if tool
                else "Document listing is not configured."
            )
            return self._result(str(answer), context=str(answer), direct=True)

        answer = ""
        context = ""
        if self.rag_graph:
            try:
                rag_input = {
                    "question": query,
                    "user_id": user_id,
                    "skip_memory_persist": True,
                    "skip_generate": True,
                    "rewritten_question": None,
                    "documents": [],
                    "context": "",
                    "answer": "",
                    "documents_relevant": False,
                    "has_error": False,
                    "retry_count": 0,
                }
                config = {
                    "configurable": {
                        "thread_id": f"{thread_id}:rag" if thread_id else "rag",
                    }
                }
                result = await self.rag_graph.ainvoke(rag_input, config=config)
                context = result.get("context", "")
                answer = result.get("answer") or result.get("generation", "")
                if context and not answer:
                    answer = "Retrieved internal documents."
            except Exception as exc:
                logger.error("RAG graph invocation failed: %s", exc, exc_info=True)

        if not answer and self.registry and user_id:
            tool = self.registry.get_tool_for_agent("rag_agent", "document_search")
            if tool:
                answer = str(
                    await invoke_tool(
                        tool,
                        {"query": query, "top_k": 4, "user_id": user_id},
                        user_id=user_id,
                        session_id=thread_id,
                    )
                )
        return self._result(
            answer or "No relevant internal documents found.",
            context=context,
            direct=not bool(context),
        )

    @staticmethod
    def _result(answer: str, *, context: str, direct: bool = False) -> dict[str, Any]:
        return {
            "messages": [
                HumanMessage(content=f"[RAG Specialist]: {answer}", name="rag_agent")
            ],
            "agent_outputs": [
                AgentOutput(
                    agent_name="rag_agent",
                    result=answer,
                    metadata={"context": context, "direct_response": direct},
                )
            ],
        }
