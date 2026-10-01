"""Laya supervisor routing without loading the real checkpoint."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from app.agent.multi_agent.supervisor import SupervisorDecision
from app.services.llm.laya import LayaService, _questions_from_model


def test_supervisor_schema_maps_to_laya_choice_question() -> None:
    questions = _questions_from_model(SupervisorDecision)
    assert "next" in questions
    assert questions["next"]["type"] == "choice"
    assert set(questions["next"]["criteria"]) == {
        "rag_agent",
        "web_agent",
        "general_agent",
        "FINISH",
    }


def test_laya_structured_route_uses_choice_not_text() -> None:
    agent = MagicMock()
    agent.predict.return_value = {
        "answers": {
            "next": {
                "type": "choice",
                "choice": "web_agent",
                "confidence": 0.91,
            }
        }
    }
    service = LayaService(agent=agent)
    decision = asyncio.run(
        service.agenerate_structured(
            prompt="User Query: latest news on TSLA",
            system_prompt="route",
            response_model=SupervisorDecision,
        )
    )
    assert decision.next == "web_agent"
    assert "web_agent" in decision.reasoning
    agent.predict.assert_called_once()
