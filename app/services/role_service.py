"""Business logic for Phase 6: Roles & Permissions.

Kept out of app/api/v1/endpoints/roles.py so the router stays a thin
request/response layer, per app/services/camera_service.py's precedent.
"""
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.db.models import EmailServer, Role, User, Zone
from app.schemas.role import RoleInput, RoleResponse
from app.services import audit_service


def _actor(user: dict) -> tuple[Optional[int], Optional[str]]:
    sub = user.get("sub")
    return (int(sub) if sub is not None else None), user.get("username")


def _to_int_id(raw: Optional[str], field_name: str) -> Optional[int]:
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Invalid {field_name}: {raw!r}")


def _to_role_response(role: Role) -> RoleResponse:
    return RoleResponse(
        id=str(role.id),
        name=role.name,
        description=role.description or "",
        permissions=role.permissions or {},
        default_email_server_id=(
            str(role.default_email_server_id) if role.default_email_server_id is not None else None
        ),
        zone_ids=[str(z) for z in (role.zone_ids or [])],
        is_system=role.is_system,
        created_at=role.created_at.isoformat(),
    )


def list_roles(db: Session, org_id: Optional[int]) -> list[RoleResponse]:
    roles = db.exec(select(Role).where(Role.org_id == org_id)).all()
    return [_to_role_response(r) for r in roles]


def get_role(db: Session, org_id: Optional[int], role_id: int) -> RoleResponse:
    role = db.get(Role, role_id)
    if role is None or role.org_id != org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Role not found.")
    return _to_role_response(role)


def user_counts(db: Session, org_id: Optional[int]) -> dict[str, int]:
    """Keyed by role id (string, matching the frontend's Record<string, number>),
    excluding soft-deleted users - mirrors the frontend mock's rolesApi.userCounts()."""
    users = db.exec(
        select(User).where(User.org_id == org_id, User.status != "soft_deleted")
    ).all()
    counts: dict[str, int] = {}
    for u in users:
        if u.role_id is None:
            continue
        key = str(u.role_id)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _validate_references(db: Session, org_id: Optional[int], payload: RoleInput) -> tuple[Optional[int], list[int]]:
    email_server_id = _to_int_id(payload.default_email_server_id, "default_email_server_id")
    if email_server_id is not None:
        server = db.get(EmailServer, email_server_id)
        if server is None or server.org_id != org_id:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "default_email_server_id does not exist.")

    zone_ids: list[int] = []
    for raw in payload.zone_ids:
        zid = _to_int_id(raw, "zone_ids")
        if zid is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "zone_ids entries cannot be empty.")
        zone = db.get(Zone, zid)
        if zone is None or zone.org_id != org_id:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"zone_ids contains an unknown zone: {raw!r}")
        zone_ids.append(zid)

    return email_server_id, zone_ids


def create_role(db: Session, user: dict, payload: RoleInput) -> RoleResponse:
    org_id = user.get("org_id")
    name = payload.name.strip()
    existing = db.exec(select(Role).where(Role.org_id == org_id)).all()
    if any(r.name.strip().lower() == name.lower() for r in existing):
        raise HTTPException(status.HTTP_409_CONFLICT, f'A role named "{name}" already exists.')

    email_server_id, zone_ids = _validate_references(db, org_id, payload)

    role = Role(
        name=name,
        description=payload.description.strip(),
        permissions=payload.permissions,
        default_email_server_id=email_server_id,
        zone_ids=zone_ids,
        is_system=False,
        org_id=org_id,
    )
    db.add(role)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, f'A role named "{name}" already exists.')
    db.refresh(role)

    actor_id, actor_name = _actor(user)
    audit_service.log_action(
        db, entity="role", entity_id=str(role.id), entity_label=role.name,
        action="create", summary=f'Created role "{role.name}".',
        actor_id=actor_id, actor_name=actor_name,
    )

    return _to_role_response(role)


def update_role(db: Session, user: dict, role_id: int, payload: RoleInput) -> RoleResponse:
    org_id = user.get("org_id")
    role = db.get(Role, role_id)
    if role is None or role.org_id != org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Role not found.")
    if role.is_system:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "The Owner role cannot be modified.")

    name = payload.name.strip()
    others = db.exec(select(Role).where(Role.org_id == org_id, Role.id != role_id)).all()
    if any(r.name.strip().lower() == name.lower() for r in others):
        raise HTTPException(status.HTTP_409_CONFLICT, f'A role named "{name}" already exists.')

    email_server_id, zone_ids = _validate_references(db, org_id, payload)

    role.name = name
    role.description = payload.description.strip()
    role.permissions = payload.permissions
    role.default_email_server_id = email_server_id
    role.zone_ids = zone_ids
    db.add(role)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, f'A role named "{name}" already exists.')
    db.refresh(role)

    actor_id, actor_name = _actor(user)
    audit_service.log_action(
        db, entity="role", entity_id=str(role.id), entity_label=role.name,
        action="update", summary=f'Updated role "{role.name}".',
        actor_id=actor_id, actor_name=actor_name,
    )

    return _to_role_response(role)


def delete_role(db: Session, user: dict, role_id: int) -> None:
    org_id = user.get("org_id")
    role = db.get(Role, role_id)
    if role is None or role.org_id != org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Role not found.")
    if role.is_system:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "The Owner role cannot be deleted.")

    has_active_users = db.exec(
        select(User).where(
            User.org_id == org_id, User.role_id == role_id, User.status != "soft_deleted"
        )
    ).first()
    if has_active_users is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "This role still has users assigned to it. Reassign those users first.",
        )

    role_name = role.name
    db.delete(role)
    db.commit()

    actor_id, actor_name = _actor(user)
    audit_service.log_action(
        db, entity="role", entity_id=str(role_id), entity_label=role_name,
        action="delete", summary=f'Deleted role "{role_name}".',
        actor_id=actor_id, actor_name=actor_name,
    )
