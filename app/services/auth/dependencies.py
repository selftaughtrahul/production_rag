"""
FastAPI dependency — extract and validate the current user from a JWT token.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError

from app.models.schemas import UserInDB
from app.services.auth.auth_service import decode_token
from database.mysql import get_db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db=Depends(get_db),
) -> UserInDB:
    """
    Decode JWT → look up user in MySQL → return UserInDB.

    Raises 401 for:
        - invalid / expired token
        - user not found
        - inactive user account
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_token(token)
        user_id: str | None = payload.get("sub")
        if not user_id:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    conn, cursor = db
    cursor.execute(
        "SELECT id, username, email, is_active FROM users WHERE id = ?",
        (user_id,),
    )
    row = cursor.fetchone()

    if row is None:
        raise credentials_exception

    if not row["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive",
        )

    return UserInDB(
        id=row["id"],
        username=row["username"],
        email=row["email"],
        is_active=row["is_active"],
    )
