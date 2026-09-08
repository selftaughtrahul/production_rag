from __future__ import annotations
from app.guardrails.base import BaseInputGuardrail, GuardrailResult
from app.guardrails.provider.nemo import NeMoProvider


class PromptInjectionGuardrail(BaseInputGuardrail):
    def __init__(self, provider: NeMoProvider):
        self.provider = provider

    async def check(self, query: str) -> GuardrailResult:
        res = await self.provider.check_prompt_injection(query)
        if not res["passed"]:
            return GuardrailResult(passed=False, reason="Prompt injection detected.", action="block")
        return GuardrailResult(passed=True, action="allow")
