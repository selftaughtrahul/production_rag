"""Cheap text checks used before calling an LLM guardrail.

Add phrases here when you see new jailbreak patterns in logs.
"""

INJECTION_HINTS = (
    "ignore previous",
    "ignore all instructions",
    "ignore your instructions",
    "you are now",
    "system prompt",
    "developer mode",
    "jailbreak",
    "override the",
    "disregard the",
    "pretend you are",
    "act as an uncensored",
    "do not follow",
)


def needs_llm_injection_check(query: str) -> bool:
    """True only when the text looks like a jailbreak. Normal questions skip NeMo."""
    text = (query or "").lower()
    return any(hint in text for hint in INJECTION_HINTS)
