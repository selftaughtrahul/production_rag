"""Audit events for agent turns and tool calls. No raw prompts or PII."""

from __future__ import annotations

from typing import Any

from app.core.logger import log_event


def audit_event(
    event: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    """Write a structured audit line (user_id, session, event name)."""
    payload = dict(data or {})
    if user_id:
        payload["user_id"] = user_id
    log_event(
        event,
        session_id=session_id,
        user_id=user_id,
        data=payload,
        message=event,
    )
