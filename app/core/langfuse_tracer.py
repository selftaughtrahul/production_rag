"""Optional Langfuse traces. Fail-open when the SDK or keys are missing."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_client: Any = None
_failed = False


def _langfuse() -> Any:
    global _client, _failed
    if _failed:
        return None
    if _client is not None:
        return _client
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    if not public_key or not secret_key:
        _failed = True
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        logger.info("langfuse package not installed; skipping traces")
        _failed = True
        return None
    _client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip(),
    )
    return _client


def trace_chat(
    *,
    user_id: str,
    session_id: str,
    outcome: str,
    agent_names: list[str] | None = None,
) -> None:
    """Record a generation span without prompt/answer PII."""
    client = _langfuse()
    if client is None:
        return
    try:
        trace = client.trace(
            name="chat",
            user_id=user_id,
            session_id=session_id,
            metadata={"outcome": outcome, "agents": agent_names or []},
        )
        trace.update(output={"outcome": outcome})
    except Exception:
        logger.warning("langfuse trace failed")
