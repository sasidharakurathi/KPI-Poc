"""Priority configuration endpoints - Phase 1.

Implements:
  GET    /api/config/priorities           List all priorities (enabled + disabled)
  POST   /api/config/priorities           Create a priority
  PUT    /api/config/priorities/{id}      Update name/color/enabled
  DELETE /api/config/priorities/{id}      Delete (only if no cameras reference it)
  PATCH  /api/config/priorities/{id}/toggle   Enable / disable

org_id is always derived from the authenticated caller (require_permission),
never from the request body - see app/schemas/config.py's module docstring.

Models used: Priority (app.db.models.domain_config)
Schemas: PriorityCreate, PriorityResponse (app.schemas.config)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.core.dependencies import DbSession, require_permission
from app.db.models.domain_config import Priority
from app.schemas.config import PriorityCreate, PriorityResponse, PriorityUpdate

router = APIRouter(prefix="/api/config/priorities", tags=["config-priorities"])


@router.get("", response_model=list[PriorityResponse], summary="List Priorities")
async def list_priorities(
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "view")),
):
    """List every priority for the caller's org, enabled or disabled - a
    disabled row must stay visible so it can be re-enabled from this list."""
    priorities = session.exec(
        select(Priority).where(Priority.org_id == user.get("org_id")).order_by(Priority.id)
    ).all()
    return priorities


@router.post("", response_model=PriorityResponse, status_code=201, summary="Create Priority")
async def create_priority(
    priority_in: PriorityCreate,
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "create")),
):
    org_id = user.get("org_id")
    existing = session.exec(
        select(Priority).where(Priority.org_id == org_id, Priority.name == priority_in.name)
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Priority with name '{priority_in.name}' already exists.")

    priority = Priority(**priority_in.model_dump(exclude_unset=True), org_id=org_id)

    session.add(priority)
    try:
        session.commit()
        session.refresh(priority)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail=f"Priority with name '{priority_in.name}' already exists.")

    return priority


@router.put("/{id}", response_model=PriorityResponse, summary="Update Priority")
async def update_priority(
    id: int,
    priority_in: PriorityUpdate,
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "edit")),
):
    org_id = user.get("org_id")
    priority = session.get(Priority, id)
    if not priority or priority.org_id != org_id:
        raise HTTPException(status_code=404, detail="Priority not found.")

    if priority_in.name is not None and priority_in.name != priority.name:
        existing = session.exec(
            select(Priority).where(Priority.org_id == org_id, Priority.name == priority_in.name)
        ).first()
        if existing:
            raise HTTPException(status_code=409, detail=f"Priority with name '{priority_in.name}' already exists.")

    update_data = priority_in.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(priority, key, value)

    session.add(priority)
    try:
        session.commit()
        session.refresh(priority)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail=f"Priority with name '{priority_in.name}' already exists.")

    return priority


@router.delete("/{id}", status_code=200, summary="Delete Priority")
async def delete_priority(
    id: int,
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "delete")),
):
    priority = session.get(Priority, id)
    if not priority or priority.org_id != user.get("org_id"):
        raise HTTPException(status_code=404, detail="Priority not found.")

    from app.db.models.camera import Camera
    in_use_by_camera = session.exec(
        select(Camera).where((Camera.priority_id == id) | (Camera.priority == priority.name))
    ).first()

    if in_use_by_camera:
        raise HTTPException(status_code=409, detail="Priority is in use by one or more cameras or KPIs.")

    session.delete(priority)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail="Database constraint violation while deleting priority.")

    return {"message": "row deleted successfully"}


@router.patch("/{id}/toggle", response_model=PriorityResponse, summary="Toggle Priority")
async def toggle_priority(
    id: int,
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "edit")),
):
    priority = session.get(Priority, id)
    if not priority or priority.org_id != user.get("org_id"):
        raise HTTPException(status_code=404, detail="Priority not found.")

    priority.enabled = not priority.enabled
    session.add(priority)
    try:
        session.commit()
        session.refresh(priority)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=400, detail="Database constraint violation while toggling priority.")

    return priority
