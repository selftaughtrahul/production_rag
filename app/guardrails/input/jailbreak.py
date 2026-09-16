"""Jailbreak detection: denylist plus NeMo when loaded (fail closed on hits)."""

from __future__ import annotations

from typing import Any

from app.guardrails.base import BaseInputGuardrail, GuardrailResult
from app.guardrails.fast_checks import is_jailbreak_denylist, needs_llm_injection_check


class JailbreakGuardrail(BaseInputGuardrail):
    def __init__(self, provider: Any | None = None) -> None:
        self.provider = provider

    async def check(self, query: str) -> GuardrailResult:
        if is_jailbreak_denylist(query):
            return GuardrailResult(
                passed=False,
                reason="Jailbreak pattern detected.",
                action="block",
            )
        if not needs_llm_injection_check(query):
            return GuardrailResult(passed=True, action="allow")
        if self.provider is None or not getattr(self.provider, "is_ready", False):
            return GuardrailResult(passed=True, action="allow")
        res = await self.provider.check_prompt_injection(query)
        if not res["passed"]:
            return GuardrailResult(
                passed=False,
                reason=str(res.get("reason") or "Jailbreak detected."),
                action="block",
            )
        return GuardrailResult(passed=True, action="allow")
