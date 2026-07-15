"""Shared FastAPI dependencies.

get_db: yields a SQLModel Session (use with Depends).
get_current_user: extracts and validates the JWT, returns the decoded payload.
  Works independently of the per-request JWTAuthMiddleware — useful when an
  endpoint needs the user identity, not just auth gating.
"""
from typing import Annotated, Generator, Optional

from fastapi import Depends, HTTPException, Request, status
from sqlmodel import Session

from app.db.engine import get_engine


def get_db() -> Generator[Session, None, None]:
    with Session(get_engine()) as session:
        yield session


DbSession = Annotated[Session, Depends(get_db)]


def get_current_user(request: Request) -> Optional[dict]:
    """Returns the JWT payload attached by JWTAuthMiddleware, or None if auth is disabled."""
    return getattr(request.state, "user", None)


def require_auth(request: Request) -> dict:
    """Dependency that requires a valid JWT. Use with Depends() on protected endpoints."""
    user = get_current_user(request)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


CurrentUser = Annotated[dict, Depends(require_auth)]
OptionalUser = Annotated[Optional[dict], Depends(get_current_user)]
