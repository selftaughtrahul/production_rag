from __future__ import annotations

from app.guardrails.base import BaseInputGuardrail, GuardrailResult
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
