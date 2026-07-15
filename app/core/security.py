"""Password hashing and verification (bcrypt).

JWT token creation/decoding lives in app.auth.jwt_utils and is re-exported
here so Phase 0 developers have a single import path.
"""
import hashlib
import secrets

import bcrypt

from app.auth.jwt_utils import (  # re-export
    InvalidTokenError,
    JWTNotConfigured,
    create_access_token,
    decode_access_token,
)

__all__ = [
    "hash_password", "verify_password",
    "generate_secure_token", "hash_token",
    "create_access_token", "decode_access_token",
    "InvalidTokenError", "JWTNotConfigured",
]


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def generate_secure_token(nbytes: int = 32) -> str:
    """Return a URL-safe random token (for refresh tokens, reset links, etc.)."""
    return secrets.token_urlsafe(nbytes)


def hash_token(raw: str) -> str:
    """SHA-256 fingerprint of a raw token — stored in DB instead of the raw value."""
    return hashlib.sha256(raw.encode()).hexdigest()
