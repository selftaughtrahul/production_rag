"""Session ownership checks against the LangGraph SQLite checkpointer."""

from __future__ import annotations

import json
import sqlite3

from fastapi import HTTPException

from database.sqlite import get_connection


def thread_owner(session_id: str) -> str | None:
    """Return the user_id stored on a checkpoint thread, if any."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT metadata FROM checkpoints WHERE thread_id = ? LIMIT 1",
            (session_id,),
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            return None
        return _owner_from_metadata(row[0])
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


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
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT thread_id, metadata FROM checkpoints GROUP BY thread_id")
        sessions: set[str] = set()
        for thread_id, metadata in cursor.fetchall():
            if _owner_from_metadata(metadata) == user_id:
                sessions.add(thread_id)
        return list(sessions)
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
