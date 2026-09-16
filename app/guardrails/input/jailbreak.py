"""Jailbreak detection: denylist plus NeMo when loaded (fail closed on hits).

Layers (edit one without touching the others):
1. denylist — always on
2. NeMo — only if the provider is ready and hints fire
"""

from __future__ import annotations

from typing import Any

from app.guardrails.base import BaseInputGuardrail, GuardrailResult
from app.guardrails.fast_checks import (
    is_jailbreak_denylist,
    needs_llm_injection_check,
    provider_is_ready,
)


def _allow(*, layer: str) -> GuardrailResult:
    return GuardrailResult(passed=True, action="allow", metadata={"layer": layer})


def _block(reason: str, *, layer: str) -> GuardrailResult:
    return GuardrailResult(
        passed=False,
        reason=reason,
        action="block",
        metadata={"layer": layer},
    )


async def check_denylist(query: str) -> GuardrailResult | None:
    """Return a block result on denylist hit; None means continue."""
    if is_jailbreak_denylist(query):
        return _block("Jailbreak pattern detected.", layer="denylist")
    return None


async def check_nemo(query: str, provider: Any | None) -> GuardrailResult:
    """Run NeMo when ready and the query looks like injection; otherwise allow."""
    if not needs_llm_injection_check(query) or not provider_is_ready(provider):
        return _allow(layer="skip_nemo")
    res = await provider.check_prompt_injection(query)
    if not res["passed"]:
        return _block(
            str(res.get("reason") or "Jailbreak detected."),
            layer="nemo",
        )
    return _allow(layer="nemo")


class JailbreakGuardrail(BaseInputGuardrail):
    def __init__(self, provider: Any | None = None) -> None:
        self.provider = provider

    async def check(self, query: str) -> GuardrailResult:
        denied = await check_denylist(query)
        if denied is not None:
            return denied
        return await check_nemo(query, self.provider)
