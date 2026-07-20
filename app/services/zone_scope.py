"""Shared zone-visibility scoping.

Used by Alerts (Phase 4), the realtime websocket, and Dashboard (Phase 5) to
restrict what a zone-restricted role (Role.zone_ids non-empty) can see.
None = unrestricted (sees everything, including camera-less alerts/totals).
A non-None set restricts to cameras in the role's zones. The built-in Owner
role and any role with an empty zone_ids list are always unrestricted.
"""
from typing import Optional

from sqlmodel import Session, select

from app.db.models import Camera, Role


def allowed_camera_ids_for_role(role: Optional[Role], db: Session) -> Optional[set[str]]:
    """Takes an already-resolved Role (or None) — for callers that need the
    role object for other checks too (e.g. the websocket endpoint's
    permission check) and don't want to look it up twice."""
    if role and role.is_system:
        return None
    zone_ids = (role.zone_ids if role else None) or []
    if not zone_ids:
        return None
    cameras = db.exec(select(Camera.camera_id).where(Camera.zone_id.in_(zone_ids))).all()
    return set(cameras)


def allowed_camera_ids_for_user(user: dict, db: Session) -> Optional[set[str]]:
    """Convenience wrapper for the require_permission dependency's user dict —
    resolves the role itself, then delegates to allowed_camera_ids_for_role."""
    if user.get("is_system"):
        return None
    role_id = user.get("role_id")
    role = db.get(Role, role_id) if role_id else None
    return allowed_camera_ids_for_role(role, db)
