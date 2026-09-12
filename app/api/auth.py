"""
Auth API — register, login, and current-user endpoints.
"""

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.schemas import RegisterRequest, TokenResponse, UserInDB
from app.services.auth.auth_service import create_access_token, hash_password, verify_password
from app.services.auth.dependencies import get_current_user
from database.models import User
from database.sqlite import get_db

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """
    Create a new user account and return a JWT access token.
    """
    if db.scalar(select(User.id).where(User.email == request.email)):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    if db.scalar(select(User.id).where(User.username == request.username)):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username is already taken",
        )

    user_id = str(uuid4())
    db.add(
        User(
            id=user_id,
            username=request.username,
            email=request.email,
            password=hash_password(request.password),
        )
    )

    token = create_access_token(user_id=user_id, username=request.username)
    return TokenResponse(access_token=token, user_id=user_id, username=request.username)


@router.post("/login", response_model=TokenResponse)
def login(
    request: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    Authenticate with email (passed as username in form) + password and return a JWT access token.
    """
    user = db.scalar(select(User).where(User.email == request.username))

    if user is None or not verify_password(request.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive",
        )

    token = create_access_token(user_id=user.id, username=user.username)
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        username=user.username,
    )


@router.get("/me", response_model=UserInDB)
def me(current_user: UserInDB = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return current_user
