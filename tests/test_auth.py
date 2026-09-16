"""Auth: password hashing and JWT round-trip."""

from __future__ import annotations

import pytest
from jose import JWTError

from app.services.auth.auth_service import (
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)


def test_hash_and_verify_password() -> None:
    pytest.importorskip("argon2")
    hashed = hash_password("s3cret")
    assert hashed != "s3cret"
    assert verify_password("s3cret", hashed) is True
    assert verify_password("wrong", hashed) is False


def test_access_token_round_trip() -> None:
    token = create_access_token(user_id="user-1", username="ada")
    payload = decode_token(token)
    assert payload["sub"] == "user-1"
    assert payload["username"] == "ada"
    assert "exp" in payload


def test_decode_rejects_garbage_token() -> None:
    with pytest.raises(JWTError):
        decode_token("not-a-jwt")
