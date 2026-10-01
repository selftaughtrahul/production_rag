"""Redis response cache keyed by user_id + query hash."""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_client: Any = None
_client_failed = False


def cache_key(user_id: str, question: str) -> str:
    digest = hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()
    return f"rag:answer:{user_id}:{digest}"


def _redis() -> Any:
    global _client, _client_failed
    if _client_failed:
        return None
    if _client is not None:
        return _client
    try:
        import redis
    except ImportError:
        _client_failed = True
        return None
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0").strip()
    try:
        client = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=0.4,
            socket_timeout=0.4,
        )
        client.ping()
        _client = client
        return _client
    except redis.RedisError:
        logger.warning("Response cache Redis is unavailable; continuing without cache")
        _client_failed = True
        return None


def get_cached_answer(user_id: str, question: str) -> str | None:
    client = _redis()
    if client is None or not user_id or not question.strip():
        return None
    try:
        value = client.get(cache_key(user_id, question))
    except Exception:
        logger.warning("Response cache get failed")
        return None
    if isinstance(value, str) and value.strip():
        return value
    return None


def set_cached_answer(user_id: str, question: str, answer: str) -> None:
    client = _redis()
    if client is None or not user_id or not question.strip() or not answer.strip():
        return
    from app.core.config import Settings

    ttl = Settings.from_environment().response_cache_ttl_seconds
    try:
        client.setex(cache_key(user_id, question), ttl, answer)
    except Exception:
        logger.warning("Response cache set failed")
