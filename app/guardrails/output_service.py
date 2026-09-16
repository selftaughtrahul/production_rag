"""Runs output rails in a fixed order. First failure wins."""

from __future__ import annotations

from app.guardrails.base import BaseOutputGuardrail, GuardrailResult, run_output_rails


class OutputGuardrailService:
    def __init__(
        self,
        schema_validator: BaseOutputGuardrail | None = None,
        grounding_checker: BaseOutputGuardrail | None = None,
        pii_detector: BaseOutputGuardrail | None = None,
        safety_checker: BaseOutputGuardrail | None = None,
    ) -> None:
        self.schema_validator = schema_validator
        self.grounding_checker = grounding_checker
        self.pii_detector = pii_detector
        self.safety_checker = safety_checker

    def _output_rails(self) -> list[tuple[str, BaseOutputGuardrail | None]]:
        return [
            ("schema_validator", self.schema_validator),
            ("grounding_checker", self.grounding_checker),
            ("pii_detector", self.pii_detector),
            ("safety_checker", self.safety_checker),
        ]

    async def validate(
        self,
        query: str,
        response: str,
        context: str | None = None,
    ) -> GuardrailResult:
        return await run_output_rails(
            self._output_rails(),
            query,
            response,
            context,
        )
