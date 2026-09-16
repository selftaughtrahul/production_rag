"""SlowAPI key function: JWT user_id when present, else IP."""

from __future__ import annotations

from starlette.requests import Request

from app.core.rate_limit import rate_limit_key
from app.services.auth.auth_service import create_access_token


def _request(headers: list[tuple[bytes, bytes]], client_host: str = "8.8.8.8") -> Request:
    return Request(
        {
            "type": "http",
            "asgi": {"spec_version": "2.3", "version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/chat/",
            "raw_path": b"/chat/",
            "query_string": b"",
            "headers": headers,
            "client": (client_host, 12345),
            "server": ("test", 80),
        }
    )


def test_rate_limit_key_uses_jwt_user_id() -> None:
    token = create_access_token(user_id="user-42", username="ada")
    req = _request([(b"authorization", f"Bearer {token}".encode())])
    assert rate_limit_key(req) == "user:user-42"


def test_rate_limit_key_falls_back_to_ip() -> None:
    req = _request([])
    assert rate_limit_key(req) == "ip:8.8.8.8"
