import asyncio

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from database.sqlite import CHECKPOINT_DB_PATH

_checkpointer: AsyncSqliteSaver | None = None
_lock = asyncio.Lock()


async def get_checkpointer() -> AsyncSqliteSaver:
    """Lazy async SQLite checkpointer shared for the process lifetime."""
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    async with _lock:
        if _checkpointer is None:
            conn = await aiosqlite.connect(CHECKPOINT_DB_PATH, timeout=30)
            await conn.execute("PRAGMA busy_timeout=30000")
            saver = AsyncSqliteSaver(conn)
            await saver.setup()
            _checkpointer = saver
    return _checkpointer
