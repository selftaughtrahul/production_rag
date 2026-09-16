"""Cheap text checks used before calling an LLM guardrail.

Extend by appending phrases to the tuples below. Do not mix the two lists:
denylist always blocks; hints only decide whether to call NeMo.
"""

from typing import Any

# High-confidence jailbreaks. Always block, even when NeMo is not loaded.
JAILBREAK_DENYLIST: tuple[str, ...] = (
    "ignore previous",
    "ignore all instructions",
    "ignore your instructions",
    "act as an uncensored",
)

# Weaker signals. Skip the LLM rail when none of these appear.
INJECTION_HINTS: tuple[str, ...] = JAILBREAK_DENYLIST + (
    "you are now",
    "system prompt",
    "developer mode",
    "jailbreak",
    "override the",
    "disregard the",
    "pretend you are",
    "do not follow",
)


def _contains_any(query: str, phrases: tuple[str, ...]) -> bool:
    text = (query or "").lower()
    return any(phrase in text for phrase in phrases)


def is_jailbreak_denylist(query: str) -> bool:
    """True for high-confidence jailbreak phrases that never need an LLM rail."""
    return _contains_any(query, JAILBREAK_DENYLIST)


def needs_llm_injection_check(query: str) -> bool:
    """True only when the text looks like a jailbreak. Normal questions skip NeMo."""
    return _contains_any(query, INJECTION_HINTS)


def provider_is_ready(provider: object | None) -> bool:
    """Shared ready check for NeMo / Llama Guard / Presidio-style providers."""
    return provider is not None and bool(getattr(provider, "is_ready", False))


async def nemo_injection_verdict(
    query: str,
    provider: Any | None,
) -> dict[str, Any] | None:
    """NeMo prompt-injection result, or None when the LLM rail should be skipped."""
    if not provider_is_ready(provider) or not needs_llm_injection_check(query):
        return None
    return await provider.check_prompt_injection(query)
