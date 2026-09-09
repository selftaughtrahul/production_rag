from __future__ import annotations

from typing import TYPE_CHECKING

from app.guardrails.base import BaseInputGuardrail, GuardrailResult

if TYPE_CHECKING:
    from app.guardrails.provider.presidio import PresidioProvider

class PresidioPIIGuardrail(BaseInputGuardrail):
    def __init__(self, provider: PresidioProvider):
        self.provider = provider

    async def check(self, query: str) -> GuardrailResult:
        res = self.provider.analyze_and_anonymize(query)
        if not res["passed"]:
            return GuardrailResult(
                passed=False,
                reason=f"PII detected: {', '.join(res['entities'])}",
                action="anonymize",
                metadata={"anonymized_text": res["anonymized_text"]}
            )
        return GuardrailResult(passed=True, action="allow")
