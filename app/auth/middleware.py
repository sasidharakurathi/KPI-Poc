from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .jwt_utils import InvalidTokenError, JWTNotConfigured, decode_access_token

_EXCLUDED_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}


class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        from app.config import settings

        if not settings.JWT_AUTH_ENABLED:
            return await call_next(request)

        if request.method == "OPTIONS" or request.url.path in _EXCLUDED_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header."},
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header[len("Bearer "):]
        try:
            payload = decode_access_token(token)
        except (InvalidTokenError, JWTNotConfigured) as e:
            return JSONResponse(
                status_code=401,
                content={"detail": str(e)},
                headers={"WWW-Authenticate": "Bearer"},
            )

        request.state.user = payload
        return await call_next(request)
