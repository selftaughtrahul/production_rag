"""
Auth service — password hashing and JWT token operations.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import Settings




_pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto",
)


def hash_password(plain: str) -> str:
    """Hash a plain-text password."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain-text password against its hash."""
    return _pwd_context.verify(plain, hashed)

def create_access_token(user_id: str, username: str) -> str:
    """
    Create a signed JWT access token.

    Payload fields:
        sub      — user_id (subject)
        username — display name
        exp      — expiry timestamp
    """
    settings = Settings.from_environment()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.jwt_expire_minutes
    )
    payload = {
        "sub": user_id,
        "username": username,
        "exp": expire,
    }
    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT token.

    Returns the payload dict.
    Raises jose.JWTError on invalid / expired tokens.
    """
    settings = Settings.from_environment()
    return jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
    )
