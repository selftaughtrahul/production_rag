"""
Auth API — register, login, and current-user endpoints.
"""

from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from app.models.schemas import (LoginRequest, RegisterRequest, TokenResponse, UserInDB)
from app.services.auth.auth_service import (create_access_token, hash_password,verify_password)
from app.services.auth.dependencies import get_current_user
from database.sqlite import get_db

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest, db=Depends(get_db)):
    """
    Create a new user account and return a JWT access token.
    """

    conn, cursor = db

    # Check for duplicate email
    cursor.execute("SELECT id FROM users WHERE email = ?",(request.email,),)

    if cursor.fetchone():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    # Check for duplicate username
    cursor.execute("SELECT id FROM users WHERE username = ?",(request.username,),)

    if cursor.fetchone():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username is already taken",
        )

    # Generate user ID
    user_id = str(uuid4())

    # Hash password
    hashed = hash_password(request.password)

    # Create user
    cursor.execute(
        """
        INSERT INTO users (
            id,
            username,
            email,
            password
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            user_id,
            request.username,
            request.email,
            hashed,
        ),
    )

    # Create JWT
    token = create_access_token(user_id=user_id,username=request.username)

    return TokenResponse(access_token=token,user_id=user_id,username=request.username)



@router.post("/login", response_model=TokenResponse)
def login(request: OAuth2PasswordRequestForm = Depends(), db=Depends(get_db)):
    """
    Authenticate with email (passed as username in form) + password and return a JWT access token.
    """
    conn, cursor = db

    cursor.execute(
        "SELECT id, username, email, password, is_active FROM users WHERE email = ?",
        (request.username,),
    )
    row = cursor.fetchone()

    if row is None or not verify_password(request.password, row["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not row["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive",
        )

    token = create_access_token(user_id=row["id"], username=row["username"])

    return TokenResponse(
        access_token=token,
        user_id=row["id"],
        username=row["username"],
    )



@router.get("/me", response_model=UserInDB)
def me(current_user: UserInDB = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return current_user
