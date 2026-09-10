from __future__ import annotations

from typing import TYPE_CHECKING

from app.guardrails.base import BaseInputGuardrail, GuardrailResult
from app.guardrails.fast_checks import needs_llm_injection_check

if TYPE_CHECKING:
    from app.guardrails.provider.nemo import NeMoProvider


class PromptInjectionGuardrail(BaseInputGuardrail):
    def __init__(self, provider: NeMoProvider):
        self.provider = provider

    async def check(self, query: str) -> GuardrailResult:
        if not getattr(self.provider, "is_ready", False):
            return GuardrailResult(passed=True, action="allow")
        # Normal questions skip a 5–9s NeMo LLM call.
        if not needs_llm_injection_check(query):
            return GuardrailResult(passed=True, action="allow")
        res = await self.provider.check_prompt_injection(query)
        if not res["passed"]:
            return GuardrailResult(passed=False, reason="Prompt injection detected.", action="block")
        return GuardrailResult(passed=True, action="allow")
