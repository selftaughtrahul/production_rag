from __future__ import annotations

from ..base import BaseInputGuardrail, GuardrailResult


class InputValidationGuardrail(BaseInputGuardrail):

    def __init__(self, min_length: int = 2,max_length: int = 10_000):
        self.min_length = min_length
        self.max_length = max_length

    async def check(self, query: str) -> GuardrailResult:

        if not isinstance(query, str):
            return GuardrailResult(
                passed=False,
                reason="Query must be a string.",
                action="block",
            )

        query = query.strip()

        if len(query) < self.min_length:
            return GuardrailResult(
                passed=False,
                reason="Query is empty or too short.",
                action="block",
            )

        if len(query) > self.max_length:
            return GuardrailResult(
                passed=False,
                reason="Query exceeds maximum allowed length.",
                action="block",
                metadata={
                    "max_length": self.max_length,
                },
            )

        return GuardrailResult(
            passed=True,
            action="allow",
        )