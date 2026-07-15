from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt


class JWTNotConfigured(RuntimeError):
    pass


class InvalidTokenError(ValueError):
    pass


def _secret() -> str:
    from app.config import settings
    if not settings.JWT_SECRET_KEY:
        raise JWTNotConfigured(
            "JWT_SECRET_KEY is not set. Generate one with: "
            "python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return settings.JWT_SECRET_KEY


def create_access_token(
    subject: str,
    roles: Optional[list[str]] = None,
    extra_claims: Optional[dict] = None,
    expires_minutes: Optional[int] = None,
) -> str:
    from app.config import settings
    now = datetime.now(timezone.utc)
    minutes = expires_minutes if expires_minutes is not None else settings.JWT_EXPIRE_MINUTES
    payload: dict = {
        "sub": subject,
        "roles": roles or [],
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
        "iss": settings.JWT_ISSUER,
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, _secret(), algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    from app.config import settings
    try:
        return jwt.decode(
            token,
            _secret(),
            algorithms=[settings.JWT_ALGORITHM],
            issuer=settings.JWT_ISSUER,
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as e:
        raise InvalidTokenError(f"Token expired: {e}") from e
    except jwt.InvalidTokenError as e:
        raise InvalidTokenError(f"Invalid token: {e}") from e
