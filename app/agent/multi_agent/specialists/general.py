"""General specialist with deterministic utility tools."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage

from app.agent.multi_agent.state import AgentOutput, SupervisorState
from app.prompts.specialist_prompts import GENERAL_SYSTEM_V1
from app.services.llm.claude import ClaudeService
from app.tools.execute import invoke_tool
from app.tools.registry import ToolRegistry

_CALC_RE = re.compile(
    r"(?:calculate|compute|what is)\s+([0-9\s+\-*/%().]{1,200})[?]?$",
    re.IGNORECASE,
)
_TIMEZONES = {
    "utc": "UTC",
    "india": "Asia/Kolkata",
    "kolkata": "Asia/Kolkata",
    "new york": "America/New_York",
    "london": "Europe/London",
    "tokyo": "Asia/Tokyo",
}


class GeneralLLMNode:
    def __init__(self, llm: ClaudeService, tool_registry: ToolRegistry) -> None:
        self.llm = llm
        self.registry = tool_registry

    async def __call__(self, state: SupervisorState) -> dict[str, Any]:
        query = state.get("query", "")
        user_id = state.get("user_id")
        session_id = state.get("thread_id")
        tool_output = await self._maybe_use_tool(query, user_id, session_id)
        memories = state.get("long_term_memories") or []
        memory_text = "\n".join(f"- {memory}" for memory in memories) or "(none)"
        response = await self.llm.agenerate(
            prompt=(
                f"User query:\n{query}\n\n"
                f"User profile facts:\n{memory_text}\n\n"
                f"Deterministic tool output:\n{tool_output or '(none)'}"
            ),
            system_prompt=GENERAL_SYSTEM_V1,
            max_tokens=1024,
        )
        return {
            "messages": [
                HumanMessage(
                    content=f"[General Specialist]: {response}",
                    name="general_agent",
                )
            ],
            "agent_outputs": [
                AgentOutput(
                    agent_name="general_agent",
                    result=response,
                    metadata={
                        "tool_used": bool(tool_output),
                        "direct_response": True,
                    },
                )
            ],
        }

    async def _maybe_use_tool(
        self,
        query: str,
        user_id: str | None,
        session_id: str | None,
    ) -> str | None:
        calculation = _CALC_RE.search(query.strip())
        if calculation:
            tool = self.registry.get_tool_for_agent("general_agent", "calculate")
            if tool:
                return str(
                    await invoke_tool(
                        tool,
                        {"expression": calculation.group(1).strip()},
                        user_id=user_id,
                        session_id=session_id,
                    )
                )
        lowered = query.lower()
        if "time" in lowered or "date" in lowered:
            timezone_name = next(
                (zone for hint, zone in _TIMEZONES.items() if hint in lowered),
                "UTC",
            )
            tool = self.registry.get_tool_for_agent(
                "general_agent",
                "current_datetime",
            )
            if tool:
                return str(
                    await invoke_tool(
                        tool,
                        {"timezone_name": timezone_name},
                        user_id=user_id,
                        session_id=session_id,
                    )
                )
        return None
