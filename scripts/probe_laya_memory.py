"""Print local Laya memory decisions for a few short-term memory states.

Run from the repo root:

    .\\venv\\Scripts\\python.exe scripts\\probe_laya_memory.py
"""

from __future__ import annotations

import asyncio
import time

from app.memory.short_term import MemoryCompressionConfig, should_compress_memory
from app.services.llm.laya import LayaService

_EXAMPLES = [
    {
        "name": "under_threshold",
        "token_count": 800,
        "message_count": 4,
        "recent_count": 4,
        "has_summary": False,
        "compression_count": 0,
        "current_task": "ask what a token is",
    },
    {
        "name": "just_over_threshold",
        "token_count": 17_200,
        "message_count": 40,
        "recent_count": 6,
        "has_summary": False,
        "compression_count": 0,
        "current_task": "explain prompt engineering from internal docs",
    },
    {
        "name": "already_summarized",
        "token_count": 17_500,
        "message_count": 12,
        "recent_count": 6,
        "has_summary": True,
        "compression_count": 1,
        "current_task": "continue the same documentation question",
    },
    {
        "name": "repetitive_long_thread",
        "token_count": 28_000,
        "message_count": 80,
        "recent_count": 6,
        "has_summary": True,
        "compression_count": 3,
        "current_task": "user keeps repeating the same order status question",
    },
    {
        "name": "hard_limit",
        "token_count": 21_000,
        "message_count": 55,
        "recent_count": 6,
        "has_summary": False,
        "compression_count": 0,
        "current_task": "compare two uploaded policy documents",
    },
]


async def main() -> None:
    router = LayaService()
    config = MemoryCompressionConfig(threshold=16_000, hard_limit=20_000, keep_recent_messages=6)
    for example in _EXAMPLES:
        name = example.pop("name")
        started = time.perf_counter()
        decision, confidence, fallback = await should_compress_memory(
            router=router,
            config=config,
            **example,
        )
        elapsed = int((time.perf_counter() - started) * 1000)
        print(
            f"{name}: decision={decision} confidence={confidence} "
            f"fallback={fallback} latency_ms={elapsed}"
        )


if __name__ == "__main__":
    asyncio.run(main())
