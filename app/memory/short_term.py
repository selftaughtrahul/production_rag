"""Short-term checkpoint compression. Laya decides; Claude writes the summary."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import BaseMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import BaseModel, Field

from app.agent.multi_agent.state import SupervisorState
from app.core.config import Settings

logger = logging.getLogger(__name__)

SUMMARY_MESSAGE_ID = "memory-summary"
DecisionName = Literal["KEEP", "COMPRESS", "AGGRESSIVE_COMPRESS"]

_DECISION_CRITERIA: dict[str, str] = {
    "KEEP": (
        "the thread is still usable, recent turns are distinct, or an existing "
        "summary already covers the older turns"
    ),
    "COMPRESS": (
        "older turns should be folded into structured memory while the latest "
        "turns stay verbatim"
    ),
    "AGGRESSIVE_COMPRESS": (
        "the thread is far over the limit, highly repetitive, or the existing "
        "summary is stale and bloated"
    ),
}

_SUMMARY_SYSTEM = """\
You compress older conversation turns into structured memory.
Do not answer the user. Do not write one undifferentiated paragraph.
Keep user facts, the current task, decisions, constraints, relevant tool
results, and unresolved questions. Drop greetings, repeated explanations,
duplicate questions, irrelevant tool output, and temporary reasoning.
Return exactly these sections:

SUMMARY:
...

IMPORTANT_FACTS:
- ...

CURRENT_TASK:
...

DECISIONS:
- ...

CONSTRAINTS:
- ...

UNRESOLVED_ITEMS:
- ...
"""


@dataclass(frozen=True, slots=True)
class MemoryCompressionConfig:
    enabled: bool = True
    threshold: int = 16000
    hard_limit: int = 20000
    keep_recent_messages: int = 6

    @classmethod
    def from_settings(cls, settings: Settings) -> "MemoryCompressionConfig":
        return cls(
            enabled=settings.memory_compression_enabled,
            threshold=settings.memory_compression_threshold,
            hard_limit=settings.memory_hard_limit,
            keep_recent_messages=settings.keep_recent_messages,
        )


class MemoryCompressionDecision(BaseModel):
    decision: Literal["KEEP", "COMPRESS", "AGGRESSIVE_COMPRESS"] = Field(
        ...,
        description="Whether short-term checkpoint memory should stay as-is or be compressed.",
    )
    reasoning: str = ""


def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def is_summary_message(message: BaseMessage) -> bool:
    if getattr(message, "id", None) == SUMMARY_MESSAGE_ID:
        return True
    extra = getattr(message, "additional_kwargs", None) or {}
    return bool(extra.get("memory_summary"))


def count_tokens(messages: list[BaseMessage]) -> int:
    """Approximate tokens with tiktoken, then a character fallback."""
    texts = [message_text(message) for message in messages]
    if not texts:
        return 0
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return sum(len(encoder.encode(text)) for text in texts)
    except Exception:
        logger.warning("tiktoken unavailable; estimating tokens from character length")
        return sum((len(text) + 3) // 4 for text in texts)


def _split_recent(messages: list[BaseMessage],keep_recent: int) -> tuple[list[BaseMessage], list[BaseMessage]]:
    keep = max(keep_recent, 0)
    if keep == 0 or len(messages) <= keep:
        return [], list(messages)
    return list(messages[:-keep]), list(messages[-keep:])


def _fallback_decision(token_count: int, hard_limit: int) -> DecisionName:
    if token_count > hard_limit:
        return "COMPRESS"
    return "KEEP"


async def should_compress_memory(
    *,
    token_count: int,
    message_count: int,
    recent_count: int,
    has_summary: bool,
    current_task: str,
    compression_count: int,
    config: MemoryCompressionConfig,
    router: Any | None,
) -> tuple[DecisionName, float | None, bool]:
    """Ask Laya only after the soft threshold. Failures use the hard limit."""
    if token_count < config.threshold:
        return "KEEP", None, False
    if router is None or not hasattr(router, "predict"):
        decision = _fallback_decision(token_count, config.hard_limit)
        logger.warning("Laya memory router unavailable; fallback decision=%s tokens=%s",decision,token_count)
        return decision, None, True

    state = (
        f"token_count={token_count}\n"
        f"threshold={config.threshold}\n"
        f"hard_limit={config.hard_limit}\n"
        f"message_count={message_count}\n"
        f"recent_messages={recent_count}\n"
        f"has_summary={has_summary}\n"
        f"compression_count={compression_count}\n"
        f"current_task={current_task[:500]}"
    )
    questions = {
        "decision": {
            "type": "choice",
            "instructions": (
                "Decide how to treat short-term conversation memory. "
                "Do not write a summary."
            ),
            "criteria": _DECISION_CRITERIA,
        }
    }
    try:
        result = await _predict(router, state, questions)
        hit = (result.get("answers") or {}).get("decision") or {}
        choice = hit.get("choice")
        confidence = hit.get("confidence")
        parsed = MemoryCompressionDecision.model_validate(
            {"decision": choice, "reasoning": ""}
        )
        if confidence is not None:
            confidence = float(confidence)
        return parsed.decision, confidence, False
    except Exception as exc:
        decision = _fallback_decision(token_count, config.hard_limit)
        logger.warning("Laya memory decision failed (%s); fallback decision=%s tokens=%s",exc,decision,token_count)
        return decision, None, True


async def _predict(router: Any, state: str, questions: dict[str, Any]) -> dict[str, Any]:
    import asyncio

    predict = router.predict
    if asyncio.iscoroutinefunction(predict):
        return await predict(state, questions)
    return await asyncio.to_thread(predict, state, questions)


def _summary_message(text: str) -> SystemMessage:
    return SystemMessage(
        content=text.strip(),
        id=SUMMARY_MESSAGE_ID,
        additional_kwargs={"memory_summary": True},
    )


def _compression_prompt(
    *,
    older: list[BaseMessage],
    current_task: str,
    aggressive: bool,
) -> str:
    prior = next((message_text(message) for message in older if is_summary_message(message)), "")
    transcript = "\n".join(
        f"{getattr(message, 'type', 'message')}: {message_text(message)}"
        for message in older
        if not is_summary_message(message)
    )
    mode = "aggressive" if aggressive else "normal"
    return (
        f"Mode: {mode}\n"
        f"Current task: {current_task or '(unknown)'}\n\n"
        f"Existing structured memory:\n{prior or '(none)'}\n\n"
        f"Older turns to fold in:\n{transcript or '(none)'}\n"
    )


async def manage_short_term_memory(
    state: SupervisorState,
    *,
    llm: Any | None,
    router: Any | None,
    config: MemoryCompressionConfig | None = None,
) -> dict[str, Any]:
    """Return a checkpoint update, or nothing when memory should stay as-is."""
    started = time.perf_counter()
    settings_config = config or MemoryCompressionConfig.from_settings(Settings.from_environment())
    if not settings_config.enabled:
        return {}

    messages = list(state.get("messages") or [])
    if not messages:
        logger.info("MEMORY: before=0 laya_decision=KEEP removed_messages=0 latency_ms=0")
        return {}

    keep_recent = settings_config.keep_recent_messages
    older, recent = _split_recent(messages, keep_recent)
    fresh_older = [message for message in older if not is_summary_message(message)]
    token_count = count_tokens(messages)
    has_summary = any(is_summary_message(message) for message in messages)
    logger.info(
        "MEMORY: before=%s messages=%s threshold=%s",
        token_count,
        len(messages),
        settings_config.threshold,
    )

    if not fresh_older and token_count <= settings_config.hard_limit:
        logger.info(
            "MEMORY: before=%s laya_decision=KEEP reason=already_compressed latency_ms=%s",
            token_count,
            int((time.perf_counter() - started) * 1000),
        )
        return {}

    if not older:
        logger.info(
            "MEMORY: before=%s laya_decision=KEEP reason=within_recent_window latency_ms=%s",
            token_count,
            int((time.perf_counter() - started) * 1000),
        )
        return {}

    decision, confidence, used_fallback = await should_compress_memory(
        token_count=token_count,
        message_count=len(messages),
        recent_count=len(recent),
        has_summary=has_summary,
        current_task=str(state.get("query") or ""),
        compression_count=int(state.get("compression_count") or 0),
        config=settings_config,
        router=router,
    )
    if decision == "KEEP":
        logger.info(
            "MEMORY: before=%s laya_decision=KEEP confidence=%s fallback=%s latency_ms=%s",
            token_count,
            confidence,
            used_fallback,
            int((time.perf_counter() - started) * 1000),
        )
        return {}

    aggressive = decision == "AGGRESSIVE_COMPRESS"
    if aggressive:
        older, recent = _split_recent(messages, max(2, keep_recent // 3))
    if not any(not is_summary_message(message) for message in older) and not aggressive:
        return {}

    if llm is None or not hasattr(llm, "agenerate"):
        logger.warning("Compression model unavailable; keeping checkpoint memory")
        return {}

    try:
        summary = await llm.agenerate(
            prompt=_compression_prompt(
                older=older,
                current_task=str(state.get("query") or ""),
                aggressive=aggressive,
            ),
            system_prompt=_SUMMARY_SYSTEM,
            max_tokens=500 if aggressive else 900,
        )
    except Exception as exc:
        logger.warning("Memory compression model failed (%s); keeping messages", exc)
        return {}

    summary = (summary or "").strip()
    if not summary:
        logger.warning("Memory compression model returned an empty summary; keeping messages")
        return {}

    updated = [_summary_message(summary), *recent]
    after_tokens = count_tokens(updated)
    removed = max(len(messages) - len(updated), 0)
    latency_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "MEMORY: before=%s laya_decision=%s confidence=%s after=%s "
        "removed_messages=%s latency_ms=%s fallback=%s",
        token_count,
        decision,
        confidence,
        after_tokens,
        removed,
        latency_ms,
        used_fallback,
    )
    return {
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *updated],
        "memory_summary": summary,
        "memory_version": int(state.get("memory_version") or 0) + 1,
        "last_compressed_message_index": len(older),
        "compression_count": int(state.get("compression_count") or 0) + 1,
    }
