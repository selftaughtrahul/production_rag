"""Session ownership checks against the LangGraph SQLite checkpointer."""

from __future__ import annotations

import json

from fastapi import HTTPException
from sqlalchemy import MetaData, Table, inspect, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from database.sqlite import SessionLocal

_checkpoints: Table | None = None


def _checkpoints_table(session: Session) -> Table | None:
    """Reflect LangGraph's checkpoints table without owning its schema."""
    global _checkpoints
    bind = session.get_bind()
    if not inspect(bind).has_table("checkpoints"):
        return None
    if _checkpoints is None:
        _checkpoints = Table("checkpoints", MetaData(), autoload_with=bind)
    return _checkpoints


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


def _owner_from_metadata(metadata) -> str | None:
    if not metadata:
        return None
    try:
        if isinstance(metadata, bytes):
            metadata = metadata.decode("utf-8")
        meta_dict = json.loads(metadata)
        return meta_dict.get("user_id")
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
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
    try:
        table = _checkpoints_table(session)
        if table is None:
            return []
        rows = session.execute(
            select(table.c.thread_id, table.c["metadata"]).group_by(table.c.thread_id)
        ).all()
        sessions: set[str] = set()
        for thread_id, metadata in rows:
            if _owner_from_metadata(metadata) == user_id:
                sessions.add(thread_id)
        return list(sessions)
    except OperationalError:
        return []
    finally:
        session.close()
