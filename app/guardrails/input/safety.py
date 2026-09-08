
# app/guardrails/input/safety.py
from app.guardrails.base import BaseInputGuardrail, GuardrailResult
from app.guardrails.provider.llama_guard import LlamaGuardProvider

class InputSafetyGuardrail(BaseInputGuardrail):
    def __init__(self, provider: LlamaGuardProvider):
        self.provider = provider

    async def check(self, query: str) -> GuardrailResult:
        res = await self.provider.check_safety(query)
        if not res["passed"]:
            return GuardrailResult(passed=False, reason="Safety policy violation.", action="block")
        return GuardrailResult(passed=True, action="allow")

