"""Shared FastAPI dependencies.

get_db: yields a SQLModel Session (use with Depends).
get_current_user: reads the JWT payload attached by JWTAuthMiddleware, if any
  — signature/expiry-valid only, no DB check (cheap, used where "is there any
  token at all" is enough, e.g. OptionalUser).
require_auth: 401s unless a valid Bearer token was presented AND the account
  is still active AND the token hasn't been revoked (see token_version below).
require_permission(module, action): 401/403s unless the caller's Role grants
  `action` on `module` (app.core.permissions vocabulary). The built-in Owner
  role (is_system=True) always passes.
"""
from typing import Annotated, Generator, Optional

from fastapi import Depends, HTTPException, Request, status
from sqlmodel import Session

from app.db.engine import get_engine
from app.db.models import Role, User


def get_db() -> Generator[Session, None, None]:
    with Session(get_engine()) as session:
        yield session


DbSession = Annotated[Session, Depends(get_db)]


def get_current_user(request: Request) -> Optional[dict]:
    """Returns the JWT payload attached by JWTAuthMiddleware, or None if no
    valid Bearer token was presented on this request."""
    return getattr(request.state, "user", None)


def require_auth(request: Request, db: DbSession) -> dict:
    """Requires a valid, unexpired JWT, AND re-checks against the DB on every
    call: the account must still be status="active", and the JWT's
    token_version claim must match the user's current token_version.

    That second check is what makes logout/password-reset/future disable-
    delete take effect immediately instead of waiting out the access token's
    natural expiry — see app.services.auth_service.revoke_all_sessions().
    This costs one extra DB lookup per authenticated request, a deliberate
    tradeoff for real revocation over a purely stateless JWT.
    """
    payload = get_current_user(request)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.get(User, int(payload["sub"]))
    if user is None or user.status != "active":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session is no longer valid.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if user.token_version != payload.get("token_version", 0):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has been revoked. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


CurrentUser = Annotated[dict, Depends(require_auth)]
OptionalUser = Annotated[Optional[dict], Depends(get_current_user)]


def require_permission(module: str, action: str):
    """Dependency factory: requires auth AND that the caller's role grants
    `action` on `module`. Every PRD management endpoint from Phase 0 onward
    should be gated with this, per "every permission rule is enforced twice
    — hidden in the UI, and rejected again on the server" (PRD Cross-Cutting
    Notes)."""

    def _dependency(request: Request, db: DbSession) -> dict:
        user = require_auth(request, db)
        if user.get("is_system"):
            return user
        role_id = user.get("role_id")
        role = db.get(Role, role_id) if role_id else None
        permissions = (role.permissions if role else None) or {}
        if action not in permissions.get(module, []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role does not have '{action}' permission on '{module}'.",
            )
        return user

    return _dependency
