"""Cheap text checks used before calling an LLM guardrail.

Add phrases here when you see new jailbreak patterns in logs.
"""

# Always block even when NeMo is not loaded. Keep this list tight — it is not
# the same as INJECTION_HINTS (those only decide whether to call the LLM rail).
JAILBREAK_DENYLIST = (
    "ignore previous",
    "ignore all instructions",
    "ignore your instructions",
    "act as an uncensored",
)

INJECTION_HINTS = JAILBREAK_DENYLIST + (
    "you are now",
    "system prompt",
    "developer mode",
    "jailbreak",
    "override the",
    "disregard the",
    "pretend you are",
    "do not follow",
)


def is_jailbreak_denylist(query: str) -> bool:
    """True for high-confidence jailbreak phrases that never need an LLM rail."""
    text = (query or "").lower()
    return any(phrase in text for phrase in JAILBREAK_DENYLIST)


def needs_llm_injection_check(query: str) -> bool:
    """True only when the text looks like a jailbreak. Normal questions skip NeMo."""
    text = (query or "").lower()
    return any(hint in text for hint in INJECTION_HINTS)
