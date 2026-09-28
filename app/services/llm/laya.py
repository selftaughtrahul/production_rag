from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from langsmith import traceable

from app.core.config import Settings
from app.core.device import resolve_torch_device
from app.prompts.specialist_prompts import SUPERVISOR_LAYA_CRITERIA, SUPERVISOR_ROUTING_V2

logger = logging.getLogger(__name__)


def _questions_from_model(response_model: type) -> dict[str, dict[str, Any]]:
    """Map Pydantic Literal/enum fields to Laya choice questions."""
    schema = response_model.model_json_schema()
    questions: dict[str, dict[str, Any]] = {}
    for name, prop in schema.get("properties", {}).items():
        enums = prop.get("enum")
        if not enums:
            continue
        criteria = {
            key: SUPERVISOR_LAYA_CRITERIA.get(str(key), str(key)) for key in enums
        }
        questions[name] = {
            "type": "choice",
            "instructions": prop.get("description") or SUPERVISOR_ROUTING_V2,
            "criteria": criteria,
        }
    if not questions:
        raise ValueError(
            "LayaService.agenerate_structured requires a Literal/enum field "
            f"on {response_model.__name__}"
        )
    return questions


def _model_from_answers(
    response_model: type,
    answers: dict[str, Any],
    reasoning: str,
):
    payload: dict[str, Any] = {}
    schema = response_model.model_json_schema()
    for name, prop in schema.get("properties", {}).items():
        hit = answers.get(name)
        if isinstance(hit, dict) and hit.get("type") == "choice":
            payload[name] = hit["choice"]
            continue
        if name == "reasoning" or prop.get("type") == "string":
            payload[name] = reasoning
    return response_model.model_validate(payload)


def _format_reasoning(answers: dict[str, Any]) -> str:
    parts: list[str] = []
    for name, hit in answers.items():
        if not isinstance(hit, dict) or hit.get("type") != "choice":
            continue
        choice = hit.get("choice")
        confidence = hit.get("confidence")
        parts.append(f"{name}={choice} confidence={confidence}")
    return "laya " + "; ".join(parts) if parts else "laya"


class LayaService:
    """Local Laya typed-decision model. Used for supervisor routing, not generation."""

    def __init__(
        self,
        model: str | None = None,
        *,
        agent: Any | None = None,
        device: str | None = None,
    ) -> None:
        os.environ.setdefault("USE_TF", "0")
        self._agent = agent
        self.model = model
        self.device = device
        if self._agent is None:
            self._agent = self._load()

    def _load(self) -> Any:
        import laya

        settings = Settings.from_environment()
        model = self.model or settings.laya_model_path or settings.laya_model
        model_path = settings.local_model_path(model)
        device = resolve_torch_device(self.device or settings.embedding_device)
        logger.info("Loading Laya checkpoint from %s on %s", model_path, device)
        return laya.load(model_path, device=device)

    def predict(self, state: str | dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        return self._agent.predict(state, questions)

    @traceable(
        name="Laya Route",
        run_type="llm",
        metadata={"ls_provider": "laya", "ls_model_name": "laya"},
    )
    async def agenerate_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: type,
        max_tokens: int = 1024,
    ):
        """Answer enum fields on ``response_model``. Non-enum strings get a confidence note."""
        del system_prompt, max_tokens
        questions = _questions_from_model(response_model)
        result = await asyncio.to_thread(self.predict, prompt, questions)
        answers = result.get("answers") or {}
        reasoning = _format_reasoning(answers)
        logger.info("Laya structured decision %s", reasoning)
        return _model_from_answers(response_model, answers, reasoning)
