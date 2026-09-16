"""Runs input rails in a fixed cheap-to-expensive order.

Add a new rail by: implement BaseInputGuardrail, pass it into __init__,
and append it to `_input_rails`. Do not replace an existing slot.
"""

from __future__ import annotations

from app.guardrails.base import BaseInputGuardrail, GuardrailResult


class InputGuardrailService:
    def __init__(
        self,
        input_validator: BaseInputGuardrail | None = None,
        jailbreak: BaseInputGuardrail | None = None,
        prompt_injection: BaseInputGuardrail | None = None,
        pii_detector: BaseInputGuardrail | None = None,
        safety_checker: BaseInputGuardrail | None = None,
    ) -> None:
        self.input_validator = input_validator
        self.jailbreak = jailbreak
        self.prompt_injection = prompt_injection
        self.pii_detector = pii_detector
        self.safety_checker = safety_checker

    def _input_rails(self) -> list[tuple[str, BaseInputGuardrail | None]]:
        return [
            ("input_validator", self.input_validator),
            ("jailbreak", self.jailbreak),
            ("prompt_injection", self.prompt_injection),
            ("pii_detector", self.pii_detector),
            ("safety_checker", self.safety_checker),
        ]

    async def validate(self, query: str) -> GuardrailResult:
        for name, guardrail in self._input_rails():
            if guardrail is None:
                continue
            result = await guardrail.check(query)
            if not result.passed:
                result.metadata.setdefault("rail", name)
                return result
        return GuardrailResult(passed=True, action="allow")
