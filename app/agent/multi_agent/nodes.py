"""Shared (non-specialist) nodes for the master graph."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from app.agent.multi_agent.state import AgentOutput, SupervisorState
from app.guardrails.input_service import InputGuardrailService
from app.guardrails.output_service import OutputGuardrailService
from app.memory.persist import persist_from_turn_background
from app.memory.service import MemoryService
from app.services.llm.claude import ClaudeService
from database.sqlite import SessionLocal

logger = logging.getLogger(__name__)


class InputGuardrailNode:
    """Execute input rails and reset turn-local state."""

    def __init__(self, guardrail_service: InputGuardrailService | None = None) -> None:
        self.guardrail_service = guardrail_service

    async def __call__(self, state: SupervisorState) -> dict[str, Any]:
        update: dict[str, Any] = {
            "agent_outputs": [AgentOutput(agent_name="__reset__", result="")],
            "iterations": 0,
            "final_response": None,
            "next_node": "",
            "input_guardrail_passed": True,
            "input_guardrail_reason": None,
        }
        if self.guardrail_service is None:
            return update
        result = await self.guardrail_service.validate(state.get("query", ""))
        if not result.passed:
            logger.warning("Input guardrail blocked request: %s", result.reason)
        update.update(
            {
                "input_guardrail_passed": result.passed,
                "input_guardrail_reason": result.reason,
                "guardrail_metadata": {
                    **state.get("guardrail_metadata", {}),
                    **result.metadata,
                    "input_action": result.action,
                },
            }
        )
        return update


class LoadMemoryNode:
    """Load tenant-scoped long-term facts into supervisor state."""

    async def __call__(self, state: SupervisorState) -> dict[str, Any]:
        user_id = state.get("user_id")
        if not user_id:
            return {"long_term_memories": []}
        with SessionLocal() as session:
            try:
                memories = MemoryService(session).get_user_memories(
                    user_id=user_id,
                    limit=10,
                )
                return {"long_term_memories": [memory.memory for memory in memories]}
            except Exception as exc:
                logger.warning("Long-term memory load failed: %s", exc)
                return {"long_term_memories": []}


class OutputGuardrailNode:
    """Validate the supervisor response before delivery."""

    def __init__(self, guardrail_service: OutputGuardrailService | None = None) -> None:
        self.guardrail_service = guardrail_service

    async def __call__(self, state: SupervisorState) -> dict[str, Any]:
        if self.guardrail_service is None:
            return {
                "output_guardrail_passed": True,
                "output_guardrail_reason": None,
            }
        final_response = state.get("final_response", "")
        context = "\n\n".join(
            f"[{output.agent_name}]: {output.result}"
            for output in state.get("agent_outputs", [])
        )
        result = await self.guardrail_service.validate(
            query=state.get("query", ""),
            response=final_response,
            context=context,
        )
        sanitized = result.metadata.get("anonymized_response", final_response)
        return {
            "output_guardrail_passed": result.passed or result.action == "anonymize",
            "output_guardrail_reason": result.reason,
            "final_response": sanitized,
            "guardrail_metadata": {
                **state.get("guardrail_metadata", {}),
                **result.metadata,
                "output_action": result.action,
            },
        }


class SaveMemoryNode:
    """Extract durable facts after a completed turn."""

    def __init__(self, llm: ClaudeService) -> None:
        self.llm = llm

    async def __call__(self, state: SupervisorState) -> dict[str, Any]:
        if state.get("pending_order_action") or any(
            output.metadata.get("operational")
            for output in state.get("agent_outputs", [])
        ):
            return {}
        persist_from_turn_background(
            self.llm,
            state.get("user_id"),
            state.get("query", ""),
            state.get("final_response", ""),
        )
        return {}


class ErrorHandlerNode:
    """Return a safe failure response."""

    async def __call__(self, state: SupervisorState) -> dict[str, Any]:
        reason = (
            state.get("input_guardrail_reason")
            or state.get("output_guardrail_reason")
            or state.get("error")
            or "Request could not be processed due to a safety policy."
        )
        response = f"I cannot complete your request. Reason: {reason}"
        return {
            "final_response": response,
            "messages": [AIMessage(content=response)],
        }
