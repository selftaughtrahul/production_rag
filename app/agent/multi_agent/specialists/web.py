"""Public-web specialist."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage

from app.agent.multi_agent.state import AgentOutput, SupervisorState
from app.prompts.specialist_prompts import WEB_SUMMARY_SYSTEM_V1
from app.services.llm.claude import ClaudeService
from app.tools.execute import invoke_tool
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
_URL_RE = re.compile(r"https?://[^\s<>\"]+")


class WebAgentNode:
    def __init__(self, llm: ClaudeService, tool_registry: ToolRegistry) -> None:
        self.llm = llm
        self.registry = tool_registry

    async def __call__(self, state: SupervisorState) -> dict[str, Any]:
        query = state.get("query", "")
        user_id = state.get("user_id")
        session_id = state.get("thread_id")
        url_match = _URL_RE.search(query)
        tool_name = "read_web_page" if url_match else "web_search"
        tool = self.registry.get_tool_for_agent("web_agent", tool_name)
        if not tool:
            answer = f"{tool_name} is not available."
            return self._result(answer)

        payload: dict[str, Any] = (
            {"url": url_match.group(0).rstrip(".,)"), "max_chars": 12_000}
            if url_match
            else {"query": query, "max_results": 5}
        )
        try:
            raw = await invoke_tool(
                tool,
                payload,
                user_id=user_id,
                session_id=session_id,
            )
            answer = await self.llm.agenerate(
                prompt=f"Question:\n{query}\n\nWeb data:\n{raw}",
                system_prompt=WEB_SUMMARY_SYSTEM_V1,
                max_tokens=600,
            )
        except Exception as exc:
            logger.error("Web specialist failed: %s", exc, exc_info=True)
            answer = f"Web information could not be retrieved: {exc}"
        return self._result(answer)

    @staticmethod
    def _result(answer: str) -> dict[str, Any]:
        return {
            "messages": [
                HumanMessage(content=f"[Web Specialist]: {answer}", name="web_agent")
            ],
            "agent_outputs": [
                AgentOutput(
                    agent_name="web_agent",
                    result=answer,
                    metadata={"direct_response": True},
                )
            ],
        }
