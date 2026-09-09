"""
Pydantic request / response schemas for the RAG API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, EmailStr, field_validator


# ─────────────────────────────────────────────────────────────
# Auth schemas
# ─────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 5:
            raise ValueError("Password must be at least 5 characters")
        return v

    @field_validator("username")
    @classmethod
    def username_min_length(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str


class UserInDB(BaseModel):
    """Representation of a user row returned from MySQL."""
    id: str
    username: str
    email: str
    is_active: bool


# ─────────────────────────────────────────────────────────────
# Query schema
# ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None


# ─────────────────────────────────────────────────────────────
# Document schema
# ─────────────────────────────────────────────────────────────

from datetime import datetime

class DocumentResponse(BaseModel):
    id: str
    user_id: str
    file_name: str
    created_at: datetime


# ─────────────────────────────────────────────────────────────
# Multi-Agent schemas
# ─────────────────────────────────────────────────────────────

class AgentQueryRequest(BaseModel):
    query: str
    session_id: str | None = None


class AgentQueryResponse(BaseModel):
    success: bool = True
    session_id: str
    query: str
    final_answer: str
    iterations: int = 0
    agent_trajectory: list[dict[str, Any]] = []
    guardrail_metadata: dict[str, Any] = {}
