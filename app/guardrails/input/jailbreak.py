"""Jailbreak detection: denylist plus NeMo when loaded (fail closed on hits).

Layers (edit one without touching the others):
1. denylist — always on
2. NeMo — only if the provider is ready and hints fire
"""

from __future__ import annotations

from typing import Any

from app.guardrails.base import BaseInputGuardrail, GuardrailResult, allow, reject
from app.guardrails.fast_checks import is_jailbreak_denylist, nemo_injection_verdict


class JailbreakGuardrail(BaseInputGuardrail):
    def __init__(self, provider: Any | None = None) -> None:
        self.provider = provider

    async def check(self, query: str) -> GuardrailResult:
        if is_jailbreak_denylist(query):
            return reject("Jailbreak pattern detected.", metadata={"layer": "denylist"})
        verdict = await nemo_injection_verdict(query, self.provider)
        if verdict is None:
            return allow(metadata={"layer": "skip_nemo"})
        if not verdict["passed"]:
            return reject(
                str(verdict.get("reason") or "Jailbreak detected."),
                metadata={"layer": "nemo"},
            )
        return allow(metadata={"layer": "nemo"})
