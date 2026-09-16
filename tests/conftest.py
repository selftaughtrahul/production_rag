"""Shared test env. JWT signing must not use a missing/short secret."""

from __future__ import annotations

import os

os.environ.setdefault(
    "JWT_SECRET_KEY",
    "test-jwt-secret-key-32-chars-minimum-xx",
)
