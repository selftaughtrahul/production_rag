"""Session ownership checks against the LangGraph SQLite checkpointer."""

from __future__ import annotations

import json

import ormsgpack
from fastapi import HTTPException
from sqlalchemy import MetaData, Table, inspect, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from database.sqlite import SessionLocal

_checkpoints: Table | None = None
_writes: Table | None = None


def _checkpoints_table(session: Session) -> Table | None:
    """Reflect LangGraph's checkpoints table without owning its schema."""
    global _checkpoints
    bind = session.get_bind()
    if not inspect(bind).has_table("checkpoints"):
        return None
    if _checkpoints is None:
        _checkpoints = Table("checkpoints", MetaData(), autoload_with=bind)
    return _checkpoints


def _writes_table(session: Session) -> Table | None:
    """Reflect LangGraph's writes table (channel values per checkpoint)."""
    global _writes
    bind = session.get_bind()
    if not inspect(bind).has_table("writes"):
        return None
    if _writes is None:
        _writes = Table("writes", MetaData(), autoload_with=bind)
    return _writes


def _decode_msgpack_str(blob: object) -> str | None:
    """Decode a checkpointer msgpack blob that stores a plain string."""
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    if not isinstance(blob, (bytes, bytearray)) or not blob:
        return None
    try:
        value = ormsgpack.unpackb(bytes(blob))
    except (TypeError, ValueError):
        return None
    if isinstance(value, str) and value.strip():
        return value
    return None


def _is_user_facing_thread(thread_id: str) -> bool:
    """Nested RAG checkpoints use ``{session_id}:rag`` and are not chat sessions."""
    return bool(thread_id) and ":" not in thread_id


def thread_owner(session_id: str) -> str | None:
    """Return the user_id stored on a checkpoint thread, if any."""
    session = SessionLocal()
    try:
        table = _checkpoints_table(session)
        if table is None:
            return None
        row = session.execute(
            select(table.c["metadata"]).where(table.c.thread_id == session_id).limit(1)
        ).first()
        if not row or not row[0]:
            return None
        return _owner_from_metadata(row[0])
    except OperationalError:
        return None
    finally:
        session.close()


def _owner_from_metadata(metadata: object) -> str | None:
    """Read user_id from checkpoint metadata (dict, JSON str, or bytes)."""
    if metadata is None:
        return None
    if isinstance(metadata, memoryview):
        metadata = metadata.tobytes()
    if isinstance(metadata, (bytes, bytearray)):
        try:
            metadata = metadata.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(metadata, str):
        if not metadata:
            return None
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return None
    if isinstance(metadata, dict):
        user_id = metadata.get("user_id")
        return user_id if isinstance(user_id, str) else None
    return None


def ensure_session_owner(session_id: str | None, user_id: str) -> None:
    """404 if the session exists and belongs to a different user. New IDs are allowed."""
    if not session_id:
        return
    owner = thread_owner(session_id)
    if owner is not None and owner != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")


def list_owned_thread_ids(user_id: str) -> list[str]:
    """Return checkpoint thread IDs owned by the given user."""
    session = SessionLocal()
    session_dict = {}
   
    try:
        table = _checkpoints_table(session)
        if table is None:
            return []
        rows = session.execute(select(table.c.thread_id, table.c["metadata"])).all()
        sessions: set[str] = set()
        for thread_id, metadata in rows:
            if _owner_from_metadata(metadata) == user_id:
                sessions.add(thread_id)
        return list(sessions)
    except OperationalError:
        return []
    finally:
        session.close()

        
def _first_query_by_thread(session: Session, thread_ids: list[str]) -> dict[str, str]:
    """Map thread_id -> first supervisor ``query`` write (the user's first message)."""
    table = _writes_table(session)
    if table is None or not thread_ids:
        return {}
    rows = session.execute(
        select(table.c.thread_id, table.c.value)
        .where(table.c.thread_id.in_(thread_ids))
        .where(table.c.channel == "query")
        .order_by(table.c.checkpoint_id.asc())
    ).all()
    first: dict[str, str] = {}
    for thread_id, blob in rows:
        if thread_id in first:
            continue
        text_value = _decode_msgpack_str(blob)
        if text_value:
            first[thread_id] = text_value
    return first


def list_owned_conversations(user_id: str) -> list[dict]:
    """Return checkpoint conversations owned by the given user."""
    session = SessionLocal()

    try:
        table = _checkpoints_table(session)

        if table is None:
            return []

        rows = session.execute(
            select(
                table.c.thread_id,
                table.c["metadata"],
            )
        ).all()

        conversations: dict[str, dict] = {}

        for thread_id, metadata in rows:
            if not _is_user_facing_thread(thread_id):
                continue
            if _owner_from_metadata(metadata) != user_id:
                continue

            conversations.setdefault(
                thread_id,
                {
                    "session_id": thread_id,
                    "first_message": None,
                },
            )

        first_queries = _first_query_by_thread(session, list(conversations.keys()))
        for thread_id, conversation in conversations.items():
            conversation["first_message"] = first_queries.get(thread_id)

        return list(conversations.values())

    except OperationalError:
        return []

    finally:
        session.close()