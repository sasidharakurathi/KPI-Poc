from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .jwt_utils import InvalidTokenError, JWTNotConfigured, decode_access_token

# Exact paths that stay reachable without a token even when JWT_AUTH_ENABLED=True.
_EXCLUDED_EXACT = {"/health", "/docs", "/redoc", "/openapi.json"}

# Path prefixes the live frontend already calls today WITHOUT an Authorization
# header (see docs/IMPLEMENTATION_PLAN.md, Context item 4). Locking these down
# would break the one part of the app that currently works end-to-end, so they
# stay excluded from the hard gate even after auth is turned on globally.
_EXCLUDED_PREFIXES = ("/api/videos/", "/api/settings/")


def _is_excluded(path: str) -> bool:
    if path in _EXCLUDED_EXACT:
        return True
    return any(path.startswith(prefix) for prefix in _EXCLUDED_PREFIXES)


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Opportunistically decodes a Bearer token on every request, populating
    ``request.state.user`` whenever a caller does send one — independent of
    JWT_AUTH_ENABLED. This lets route-level ``Depends(require_auth)`` /
    ``Depends(require_permission(...))`` work correctly (e.g. from curl,
    Swagger, or pytest) even while JWT_AUTH_ENABLED stays False globally.

    JWT_AUTH_ENABLED only controls whether a *missing* token is hard-rejected
    here, for paths outside the excluded set. It is kept False by default so
    the frontend's still-unauthenticated real calls (videos/settings/health)
    are not locked out — per-route dependencies are what actually protect the
    new Phase 0+ endpoints in the meantime.
    """

    async def dispatch(self, request: Request, call_next):
        from app.config import settings

        request.state.user = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[len("Bearer "):]
            try:
                request.state.user = decode_access_token(token)
            except (InvalidTokenError, JWTNotConfigured):
                request.state.user = None

        if (
            settings.JWT_AUTH_ENABLED
            and request.method != "OPTIONS"
            and not _is_excluded(request.url.path)
            and request.state.user is None
        ):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header."},
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)
