"""Real-time alerts WebSocket — Phase 4.

Implements:
  WS /api/ws/alerts?token=<access_token>

Browsers can't set an Authorization header on a WebSocket handshake, so the
access token travels as a query parameter instead — validated the same way
require_auth validates it for REST (decode + DB status/token_version check),
plus the equivalent of require_permission("alerts", "view") against the
caller's role. Zone-scoping matches the REST endpoints exactly: a role with
a non-empty zone_ids restriction only receives events for cameras in those
zones (see app.services.zone_scope, shared with the REST Alerts/Dashboard
endpoints).

Event contract (for the frontend team):
  {"event": "alert.created",  "data": <same shape as AlertResponse from GET /api/alerts/{id}>}
  {"event": "camera.offline", "data": {"camera_id": str, "camera_name": str, "connectivity_status": "inactive"}}
  {"event": "camera.online",  "data": {"camera_id": str, "camera_name": str, "connectivity_status": "active"}}

This is a push-only channel — the server never expects the client to send
anything meaningful. The connection is closed with code 1008 (policy
violation) if the token is missing, invalid, expired, revoked, or belongs to
a role without alerts.view permission.
"""
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from sqlmodel import Session

from app.auth.jwt_utils import InvalidTokenError, JWTNotConfigured, decode_access_token
from app.db.engine import get_engine
from app.db.models import Role, User
from app.services.ws_manager import ws_manager
from app.services.zone_scope import allowed_camera_ids_for_role

router = APIRouter(tags=["realtime"])


def _authenticate(db: Session, token: str) -> tuple[Optional[User], Optional[Role]]:
    try:
        payload = decode_access_token(token)
    except (InvalidTokenError, JWTNotConfigured):
        return None, None

    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError):
        return None, None

    user = db.get(User, user_id)
    if user is None or user.status != "active":
        return None, None
    if user.token_version != payload.get("token_version", 0):
        return None, None

    role = db.get(Role, user.role_id) if user.role_id else None
    return user, role


def _has_alerts_view_permission(role: Optional[Role]) -> bool:
    if role and role.is_system:
        return True
    permissions = (role.permissions if role else None) or {}
    return "view" in permissions.get("alerts", [])


@router.websocket("/api/ws/alerts")
async def alerts_websocket(websocket: WebSocket, token: str):
    with Session(get_engine()) as db:
        user, role = _authenticate(db, token)
        if user is None or not _has_alerts_view_permission(role):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid, expired, or unauthorized token.")
            return
        allowed_camera_ids = allowed_camera_ids_for_role(role, db)

    await websocket.accept()
    conn = await ws_manager.connect(websocket, allowed_camera_ids)
    try:
        while True:
            # Push-only — nothing meaningful is expected from the client.
            # Just block here so a disconnect is noticed promptly.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(conn)
