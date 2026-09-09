from __future__ import annotations

from typing import TYPE_CHECKING

from app.guardrails.base import BaseOutputGuardrail, GuardrailResult

if TYPE_CHECKING:
    from app.guardrails.provider.presidio import PresidioProvider

class OutputPIIGuardrail(BaseOutputGuardrail):
    def __init__(self, provider: PresidioProvider):
        self.provider = provider

    async def check(self, query: str, response: str, context: str | None = None) -> GuardrailResult:
        res = self.provider.analyze_and_anonymize(response)
        if not res["passed"]:
            return GuardrailResult(
                passed=False,
                reason=f"PII leak detected in response: {', '.join(set(res['entities']))}",
                action="anonymize",
                metadata={"anonymized_response": res["anonymized_text"]}
            )
        return GuardrailResult(passed=True, action="allow")
