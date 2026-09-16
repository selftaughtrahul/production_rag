from __future__ import annotations

from typing import TYPE_CHECKING

from app.guardrails.base import BaseInputGuardrail, GuardrailResult, allow, reject
from app.guardrails.fast_checks import nemo_injection_verdict

if TYPE_CHECKING:
    from app.guardrails.provider.nemo import NeMoProvider


class PromptInjectionGuardrail(BaseInputGuardrail):
    def __init__(self, provider: NeMoProvider) -> None:
        self.provider = provider

    async def check(self, query: str) -> GuardrailResult:
        verdict = await nemo_injection_verdict(query, self.provider)
        if verdict is None:
            return allow()
        if not verdict["passed"]:
            return reject(str(verdict.get("reason") or "Prompt injection detected."))
        return allow()
