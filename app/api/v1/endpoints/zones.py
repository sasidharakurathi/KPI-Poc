"""Zone configuration endpoints — Phase 1.

Implements:
  GET    /api/config/zones
  GET    /api/config/zones/camera-counts   {zone_id: camera_count}, registered
                                            before /{id} so it isn't shadowed
  POST   /api/config/zones
  PUT    /api/config/zones/{id}
  DELETE /api/config/zones/{id}   (only when no cameras reference it)
  PATCH  /api/config/zones/{id}/toggle

org_id is always derived from the authenticated caller (require_permission),
never from the request body — see app/schemas/config.py's module docstring.

Models used: Zone (app.db.models.domain_config)
Schemas: ZoneCreate, ZoneResponse (app.schemas.config)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from app.core.dependencies import DbSession, require_permission
from app.db.models import Organization
from app.db.models.domain_config import Zone
from app.schemas.config import ZoneCreate, ZoneResponse, ZoneUpdate
from app.services.auth_service import get_timezone_or_422

router = APIRouter(prefix="/api/config/zones", tags=["config-zones"])


def _to_zone_response(zone: Zone) -> ZoneResponse:
    return ZoneResponse(
        id=zone.id,
        name=zone.name,
        timezone_id=str(zone.timezone_id) if zone.timezone_id is not None else None,
        description=zone.description,
        org_id=zone.org_id,
        enabled=zone.enabled,
        created_at=zone.created_at,
    )


def _resolve_timezone_id(session: DbSession, org_id: int, timezone_id_raw: str | None) -> int | None:
    """Validates an explicitly-given timezone_id, or falls back to the
    organization's own default_timezone_id when none is given."""
    if timezone_id_raw is not None:
        return get_timezone_or_422(session, timezone_id_raw).id
    org = session.get(Organization, org_id)
    return org.default_timezone_id if org else None


@router.get("", response_model=list[ZoneResponse], summary="List Zones")
async def list_zones(
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "view")),
):
    """List every zone for the caller's org, enabled or disabled."""
    zones = session.exec(
        select(Zone).where(Zone.org_id == user.get("org_id")).order_by(Zone.id)
    ).all()
    return [_to_zone_response(zone) for zone in zones]


@router.get("/camera-counts", response_model=dict[str, int], summary="Zone Camera Counts")
async def zone_camera_counts(
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "view")),
):
    """Camera count per zone, for the caller's org — Record<zone_id, count>,
    keyed by zone_id as a string. Zones with no cameras simply don't appear
    (matching the frontend's own accumulate-as-you-go mock behavior); cameras
    with no zone assigned yet aren't counted anywhere."""
    from app.db.models.camera import Camera

    cameras = session.exec(
        select(Camera).where(Camera.org_id == user.get("org_id"), Camera.zone_id.is_not(None))
    ).all()
    counts: dict[str, int] = {}
    for cam in cameras:
        key = str(cam.zone_id)
        counts[key] = counts.get(key, 0) + 1
    return counts


@router.post("", response_model=ZoneResponse, status_code=201, summary="Create Zone")
async def create_zone(
    zone_in: ZoneCreate,
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "create")),
):
    org_id = user.get("org_id")
    existing = session.exec(
        select(Zone).where(Zone.org_id == org_id, Zone.name == zone_in.name)
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Zone with name '{zone_in.name}' already exists.")

    zone_data = zone_in.model_dump(exclude_unset=True)
    timezone_id_raw = zone_data.pop("timezone_id", None)
    zone = Zone(
        **zone_data,
        org_id=org_id,
        timezone_id=_resolve_timezone_id(session, org_id, timezone_id_raw),
    )

    session.add(zone)
    try:
        session.commit()
        session.refresh(zone)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail=f"Zone with name '{zone_in.name}' already exists.")

    return _to_zone_response(zone)


@router.put("/{id}", response_model=ZoneResponse, summary="Update Zone")
async def update_zone(
    id: int,
    zone_in: ZoneUpdate,
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "edit")),
):
    org_id = user.get("org_id")
    zone = session.get(Zone, id)
    if not zone or zone.org_id != org_id:
        raise HTTPException(status_code=404, detail="Zone not found.")

    if zone_in.name is not None and zone_in.name != zone.name:
        existing = session.exec(
            select(Zone).where(Zone.org_id == org_id, Zone.name == zone_in.name)
        ).first()
        if existing:
            raise HTTPException(status_code=409, detail=f"Zone with name '{zone_in.name}' already exists.")

    update_data = zone_in.model_dump(exclude_unset=True)
    if "timezone_id" in update_data:
        raw = update_data.pop("timezone_id")
        update_data["timezone_id"] = get_timezone_or_422(session, raw).id if raw is not None else None
    for key, value in update_data.items():
        setattr(zone, key, value)

    session.add(zone)
    try:
        session.commit()
        session.refresh(zone)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail=f"Zone with name '{zone_in.name}' already exists.")

    return _to_zone_response(zone)


@router.delete("/{id}", status_code=200, summary="Delete Zone")
async def delete_zone(
    id: int,
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "delete")),
):
    zone = session.get(Zone, id)
    if not zone or zone.org_id != user.get("org_id"):
        raise HTTPException(status_code=404, detail="Zone not found.")

    from app.db.models.camera import Camera
    in_use_by_camera = session.exec(
        select(Camera).where((Camera.zone_id == id) | (Camera.zone == zone.name))
    ).first()

    if in_use_by_camera:
        raise HTTPException(status_code=409, detail="Zone is in use by one or more cameras.")

    session.delete(zone)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail="Database constraint violation while deleting zone.")

    return {"message": "row deleted successfully"}


@router.patch("/{id}/toggle", response_model=ZoneResponse, summary="Toggle Zone")
async def toggle_zone(
    id: int,
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "edit")),
):
    zone = session.get(Zone, id)
    if not zone or zone.org_id != user.get("org_id"):
        raise HTTPException(status_code=404, detail="Zone not found.")

    zone.enabled = not zone.enabled
    session.add(zone)
    try:
        session.commit()
        session.refresh(zone)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=400, detail="Database constraint violation while toggling zone.")

    return _to_zone_response(zone)
