from __future__ import annotations

from typing import TYPE_CHECKING

from app.guardrails.base import BaseOutputGuardrail, GuardrailResult

if TYPE_CHECKING:
    from app.guardrails.provider.llama_guard import LlamaGuardProvider

class OutputSafetyGuardrail(BaseOutputGuardrail):
    def __init__(self, provider: LlamaGuardProvider):
        self.provider = provider

    async def check(self, query: str, response: str, context: str | None = None) -> GuardrailResult:
        res = await self.provider.check_output_safety(query=query, response=response)
        if not res["passed"]:
            return GuardrailResult(
                passed=False,
                reason="Generated response violates safety policy.",
                action="block",
                metadata={"llama_guard_raw": res["raw"]}
            )
        return GuardrailResult(passed=True, action="allow")
