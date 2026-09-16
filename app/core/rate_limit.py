"""SlowAPI limiter keyed by JWT user_id, else client IP."""

from __future__ import annotations

from fastapi import FastAPI, Request
from jose import JWTError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.core.config import Settings
from app.services.auth.auth_service import decode_token

_settings = Settings.from_environment()
CHAT_LIMIT = _settings.rate_limit_chat


def rate_limit_key(request: Request) -> str:
    """Throttle authenticated calls per user_id; anonymous calls per IP."""
    header = request.headers.get("Authorization") or ""
    scheme, _, token = header.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        try:
            payload = decode_token(token.strip())
            user_id = payload.get("sub")
            if user_id:
                return f"user:{user_id}"
        except JWTError:
            pass
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(
    key_func=rate_limit_key,
    default_limits=[_settings.rate_limit_default],
    headers_enabled=True,
)


def attach_rate_limiter(app: FastAPI) -> None:
    """Register SlowAPI on the FastAPI app (default limit + 429 handler)."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
