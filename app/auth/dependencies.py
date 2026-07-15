from typing import Optional

from fastapi import HTTPException, Request, status

from .jwt_utils import InvalidTokenError, decode_access_token


def get_current_user(request: Request) -> Optional[dict]:
    return getattr(request.state, "user", None)


def require_roles(*allowed_roles: str):
    def _dep(request: Request) -> dict:
        user = get_current_user(request)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")
        user_roles: list[str] = user.get("roles", [])
        if allowed_roles and not any(r in user_roles for r in allowed_roles):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions.")
        return user
    return _dep
