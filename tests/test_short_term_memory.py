"""Short-term compression decisions without loading Laya or Claude."""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage

from app.memory.short_term import (
    MemoryCompressionConfig,
    is_summary_message,
    manage_short_term_memory,
    should_compress_memory,
)


class _Laya:
    def __init__(self, choice: str = "KEEP", fail: bool = False) -> None:
        self.choice = choice
        self.fail = fail
        self.calls = 0

    def predict(self, state: str, questions: dict) -> dict:
        del state, questions
        self.calls += 1
        if self.fail:
            raise RuntimeError("laya unavailable")
        return {
            "answers": {
                "decision": {
                    "type": "choice",
                    "choice": self.choice,
                    "confidence": 0.91,
                }
            }
        }


class _LLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def agenerate(self, prompt: str, system_prompt: str | None = None, max_tokens: int = 1024) -> str:
        del system_prompt, max_tokens
        self.prompts.append(prompt)
        return (
            "SUMMARY:\nOlder turns folded once.\n\n"
            "IMPORTANT_FACTS:\n- user prefers Python\n\n"
            "CURRENT_TASK:\nship compression\n\n"
            "DECISIONS:\n- keep recent turns verbatim\n\n"
            "CONSTRAINTS:\n- do not drop the current task\n\n"
            "UNRESOLVED_ITEMS:\n- none\n"
        )


def _config(**overrides: object) -> MemoryCompressionConfig:
    values = {
        "enabled": True,
        "threshold": 1,
        "hard_limit": 50_000,
        "keep_recent_messages": 6,
    }
    values.update(overrides)
    return MemoryCompressionConfig(**values)  # type: ignore[arg-type]


def _messages(count: int, text: str = "turn") -> list[HumanMessage]:
    return [HumanMessage(content=f"{text} {index}", id=f"m-{index}") for index in range(count)]


def _stored(update: dict) -> list:
    return [message for message in update.get("messages", []) if not isinstance(message, RemoveMessage)]


def test_below_threshold_does_not_call_laya() -> None:
    laya = _Laya(choice="COMPRESS")

    async def run() -> None:
        decision, confidence, fallback = await should_compress_memory(
            token_count=80,
            message_count=4,
            recent_count=4,
            has_summary=False,
            current_task="hello",
            compression_count=0,
            config=_config(threshold=16_000),
            router=laya,
        )
        assert decision == "KEEP"
        assert confidence is None
        assert fallback is False
        assert laya.calls == 0

    asyncio.run(run())


def test_keep_leaves_memory_unchanged() -> None:
    laya = _Laya(choice="KEEP")
    llm = _LLM()
    messages = _messages(10)

    async def run() -> None:
        update = await manage_short_term_memory(
            {"messages": messages, "query": "ship compression"},
            llm=llm,
            router=laya,
            config=_config(),
        )
        assert update == {}
        assert llm.prompts == []
        assert laya.calls == 1

    asyncio.run(run())


def test_compress_folds_older_messages_and_keeps_recent() -> None:
    laya = _Laya(choice="COMPRESS")
    llm = _LLM()
    messages = _messages(10, text="user prefers Python and asked about")

    async def run() -> None:
        update = await manage_short_term_memory(
            {"messages": messages, "query": "ship compression", "compression_count": 0},
            llm=llm,
            router=laya,
            config=_config(),
        )
        stored = _stored(update)
        assert is_summary_message(stored[0])
        assert "IMPORTANT_FACTS" in stored[0].content
        assert "CURRENT_TASK" in stored[0].content
        assert [message.content for message in stored[1:]] == [message.content for message in messages[-6:]]
        assert "user prefers Python" in llm.prompts[0]
        assert "ship compression" in llm.prompts[0]
        assert update["compression_count"] == 1
        assert update["memory_version"] == 1

    asyncio.run(run())


def test_aggressive_compress_keeps_fewer_recent_messages() -> None:
    laya = _Laya(choice="AGGRESSIVE_COMPRESS")
    llm = _LLM()
    messages = _messages(10)

    async def run() -> None:
        update = await manage_short_term_memory(
            {"messages": messages, "query": "ship compression"},
            llm=llm,
            router=laya,
            config=_config(keep_recent_messages=6),
        )
        stored = _stored(update)
        assert len(stored) == 3
        assert [message.content for message in stored[1:]] == [message.content for message in messages[-2:]]

    asyncio.run(run())


def test_laya_failure_uses_hard_limit_fallback() -> None:
    laya = _Laya(fail=True)
    llm = _LLM()
    messages = _messages(10)

    async def run() -> None:
        kept = await manage_short_term_memory(
            {"messages": messages, "query": "task"},
            llm=llm,
            router=laya,
            config=_config(threshold=1, hard_limit=100_000),
        )
        assert kept == {}
        assert llm.prompts == []

        compressed = await manage_short_term_memory(
            {"messages": messages, "query": "task"},
            llm=llm,
            router=laya,
            config=_config(threshold=1, hard_limit=1),
        )
        assert _stored(compressed)
        assert "SUMMARY:" in _stored(compressed)[0].content

    asyncio.run(run())


def test_repeated_compression_does_not_duplicate_the_summary() -> None:
    laya = _Laya(choice="COMPRESS")
    llm = _LLM()
    messages = _messages(10)

    async def run() -> None:
        first = await manage_short_term_memory(
            {"messages": messages, "query": "ship compression"},
            llm=llm,
            router=laya,
            config=_config(threshold=1, hard_limit=100_000),
        )
        stored = _stored(first)
        second = await manage_short_term_memory(
            {
                "messages": stored,
                "query": "ship compression",
                "memory_summary": first["memory_summary"],
                "memory_version": first["memory_version"],
                "compression_count": first["compression_count"],
            },
            llm=llm,
            router=laya,
            config=_config(threshold=1, hard_limit=100_000),
        )
        assert second == {}
        assert laya.calls == 1
        assert sum(isinstance(message, SystemMessage) for message in stored) == 1
        assert stored[0].content.count("SUMMARY:") == 1

    asyncio.run(run())


def test_empty_conversation() -> None:
    async def run() -> None:
        update = await manage_short_term_memory(
            {"messages": [], "query": ""},
            llm=_LLM(),
            router=_Laya(choice="COMPRESS"),
            config=_config(),
        )
        assert update == {}

    asyncio.run(run())


def test_large_conversation_keeps_recent_window() -> None:
    laya = _Laya(choice="COMPRESS")
    llm = _LLM()
    messages = _messages(400, text="detail")

    async def run() -> None:
        update = await manage_short_term_memory(
            {"messages": messages, "query": "ship compression"},
            llm=llm,
            router=laya,
            config=_config(),
        )
        stored = _stored(update)
        assert len(stored) == 7
        assert stored[-1].content == messages[-1].content
        assert "IMPORTANT_FACTS" in stored[0].content

    asyncio.run(run())
