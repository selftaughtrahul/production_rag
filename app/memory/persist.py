"""Shared long-term memory persistence used by RAG and multi-agent graphs."""

from __future__ import annotations

import logging

from app.memory.extractor import MemoryExtractor
from app.memory.service import MemoryService
from database.sqlite import get_connection

logger = logging.getLogger(__name__)


def persist_from_turn(llm, user_id: str | None, user_text: str, assistant_text: str) -> None:
    """Extract and store durable facts from a single user/assistant turn."""
    if not user_id or not user_text or not assistant_text:
        return

    conversation = f"User: {user_text}\nAssistant: {assistant_text}"
    conn = get_connection()
    try:
        memory_service = MemoryService(conn)
        existing = memory_service.get_user_memories(user_id=user_id, limit=20)
        decision = MemoryExtractor(llm).decide(
            user_id=user_id,
            conversation=conversation,
            existing_memories=existing,
        )

        if decision.action == "ADD" and decision.memory:
            memory_service.create_memory(
                user_id=user_id,
                memory=decision.memory,
                memory_type=decision.memory_type or "general",
                importance=decision.importance,
            )
            logger.info("Saved long-term memory for user=%s action=ADD", user_id)
        elif decision.action == "UPDATE" and decision.memory and decision.memory_id:
            memory_service.update_memory(
                memory_id=decision.memory_id,
                memory=decision.memory,
                memory_type=decision.memory_type or "general",
                importance=decision.importance,
                user_id=user_id,
            )
            logger.info("Updated long-term memory for user=%s action=UPDATE", user_id)
    except Exception:
        logger.warning("Failed to persist long-term memory for user=%s", user_id, exc_info=True)
    finally:
        conn.close()
