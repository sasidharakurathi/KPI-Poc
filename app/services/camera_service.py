"""Business logic for Phase 2: Camera Management.

Kept out of app/api/v1/endpoints/cameras.py so the router stays a thin
request/response layer, matching the Phase 0/6/7 service-layer pattern.

Scope note: kpi_ids stays the existing numeric list that the real detection
pipeline reads directly (app.kpis.registry) — the frontend's string-keyed
kpi_model_ids references Phase 3's KPI Management capability table, which
doesn't exist yet. Bulk endpoints, the camera-offline heartbeat monitor, and
audit-log wiring are deferred to when Phases 3/4/8 land.
"""
from datetime import datetime
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.config_loader import get_kpi_registry
from app.db.models.camera import Camera
from app.db.models.domain_config import Priority, Zone
from app.email_crypto import EmailCryptoNotConfigured, encrypt_secret
from app.kpis import list_registered_names
from app.schemas.camera import (
    CameraCreate, CameraKPIDetail, CameraListItem,
    CameraListResponse, CameraResponse, CameraUpdate,
)
from app.services import audit_service
from app.stream_recorder import stream_recorder_manager


def _actor(user: dict) -> tuple[Optional[int], Optional[str]]:
    sub = user.get("sub")
    return (int(sub) if sub is not None else None), user.get("username")

_KPI_LABELS: dict[int, str] = {
    1: "Unauthorized Access",          2: "ANPR Detection",
    3: "Intrusion Detection",          4: "Vehicle Detection",
    5: "Guard Tour / Hand Detection",  6: "Industrial PPE",
    7: "Fire / Smoke Detection",       8: "Abandoned Object",
    9: "Missing Object",               10: "Intrusion",
    11: "People Count",                12: "Engagement Time / Unattended Guest",
    13: "Staff Absence",               14: "Uniform Detection",
    15: "Carton / Box Detection",      16: "Occupancy",
    17: "Usage Time",                  18: "Fallen Object",
    19: "Guard Presence",              20: "Falling Pose",
    21: "Camera Tampering",            22: "Loading / Unloading Detection",
    23: "Cleaning as per Schedule",    24: "People Density Measurement",
    25: "Opening / Closing Time",      26: "Pack Count",
    27: "Occupancy Count & Dwell Time",28: "Detect Object",
    29: "Smoking",                     30: "Floating Object / Man Overboard",
}


def _to_int_id(raw: Optional[str], field_name: str) -> Optional[int]:
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Invalid {field_name}: {raw!r}")


def _resolve_zone_or_422(db: Session, org_id: Optional[int], raw: str) -> Zone:
    zone_id = _to_int_id(raw, "zone_id")
    zone = db.get(Zone, zone_id) if zone_id is not None else None
    if zone is None or zone.org_id != org_id:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "zone_id does not reference a known zone in your organization.")
    return zone


def _resolve_priority_or_422(db: Session, org_id: Optional[int], raw: str) -> Priority:
    priority_id = _to_int_id(raw, "priority_id")
    priority = db.get(Priority, priority_id) if priority_id is not None else None
    if priority is None or priority.org_id != org_id:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "priority_id does not reference a known priority in your organization.")
    return priority


def _kpi_details(kpi_ids: list[int]) -> list[CameraKPIDetail]:
    registry = get_kpi_registry()
    implemented_names = set(list_registered_names())
    return [
        CameraKPIDetail(
            kpi_id=kid,
            kpi_label=_KPI_LABELS.get(kid, f"KPI {kid}"),
            implemented=registry.get(str(kid)) in implemented_names,
        )
        for kid in kpi_ids
    ]


def _to_camera_response(db: Session, cam: Camera) -> CameraResponse:
    zone = db.get(Zone, cam.zone_id) if cam.zone_id is not None else None
    priority = db.get(Priority, cam.priority_id) if cam.priority_id is not None else None
    stream_status = stream_recorder_manager.status_for(cam.camera_id)
    return CameraResponse(
        camera_id=cam.camera_id,
        name=cam.name,
        zone_id=str(cam.zone_id) if cam.zone_id is not None else None,
        zone_name=zone.name if zone else None,
        priority_id=str(cam.priority_id) if cam.priority_id is not None else None,
        priority_name=priority.name if priority else None,
        priority_color=priority.color if priority else None,
        priority_level=priority.level if priority else None,
        kpi_ids=cam.kpi_ids,
        kpis=_kpi_details(cam.kpi_ids),
        status="active" if cam.enabled else "inactive",
        camera_ip=cam.camera_ip,
        rtsp_port=cam.rtsp_port,
        stream_username=cam.stream_username,
        stream_path=cam.stream_path,
        stream_password_set=bool(cam.stream_password_encrypted),
        recording_enabled=cam.recording_enabled,
        stream_status=stream_status["status"],
        stream_error=stream_status["error"],
        created_at=cam.created_at.isoformat(),
    )


def _to_camera_list_item(
    cam: Camera, zones_by_id: dict[int, Zone], priorities_by_id: dict[int, Priority],
) -> CameraListItem:
    registry = get_kpi_registry()
    implemented_names = set(list_registered_names())
    zone = zones_by_id.get(cam.zone_id) if cam.zone_id is not None else None
    priority = priorities_by_id.get(cam.priority_id) if cam.priority_id is not None else None
    stream_status = stream_recorder_manager.status_for(cam.camera_id)["status"]
    return CameraListItem(
        camera_id=cam.camera_id,
        name=cam.name,
        zone_id=str(cam.zone_id) if cam.zone_id is not None else None,
        zone_name=zone.name if zone else None,
        priority_id=str(cam.priority_id) if cam.priority_id is not None else None,
        priority_name=priority.name if priority else None,
        priority_color=priority.color if priority else None,
        priority_level=priority.level if priority else None,
        total_kpis=len(cam.kpi_ids),
        implemented_kpis=sum(1 for kid in cam.kpi_ids if registry.get(str(kid)) in implemented_names),
        status="active" if cam.enabled else "inactive",
        recording_enabled=cam.recording_enabled,
        stream_status=stream_status,
        created_at=cam.created_at.isoformat(),
    )


def list_cameras(db: Session, org_id: Optional[int]) -> CameraListResponse:
    cams = db.exec(select(Camera).where(Camera.org_id == org_id).order_by(Camera.camera_id)).all()

    zone_ids = {c.zone_id for c in cams if c.zone_id is not None}
    priority_ids = {c.priority_id for c in cams if c.priority_id is not None}
    zones_by_id = {z.id: z for z in db.exec(select(Zone).where(Zone.id.in_(zone_ids))).all()} if zone_ids else {}
    priorities_by_id = (
        {p.id: p for p in db.exec(select(Priority).where(Priority.id.in_(priority_ids))).all()}
        if priority_ids else {}
    )

    items = [_to_camera_list_item(cam, zones_by_id, priorities_by_id) for cam in cams]
    return CameraListResponse(count=len(items), cameras=items)


def get_camera(db: Session, org_id: Optional[int], camera_id: str) -> CameraResponse:
    cam = db.get(Camera, camera_id)
    if cam is None or cam.org_id != org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Camera '{camera_id}' not found.")
    return _to_camera_response(db, cam)


def create_camera(db: Session, user: dict, payload: CameraCreate) -> CameraResponse:
    org_id = user.get("org_id")
    if db.get(Camera, payload.camera_id):
        raise HTTPException(status.HTTP_409_CONFLICT, f"Camera '{payload.camera_id}' already exists.")

    zone = _resolve_zone_or_422(db, org_id, payload.zone_id)
    priority = _resolve_priority_or_422(db, org_id, payload.priority_id)

    encrypted_pw = None
    if payload.stream_password:
        try:
            encrypted_pw = encrypt_secret(payload.stream_password)
        except EmailCryptoNotConfigured as e:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))

    cam = Camera(
        camera_id=payload.camera_id,
        name=payload.name,
        zone_id=zone.id,
        priority_id=priority.id,
        kpi_ids=payload.kpi_ids,
        camera_ip=payload.camera_ip,
        rtsp_port=payload.rtsp_port,
        stream_username=payload.stream_username,
        stream_password_encrypted=encrypted_pw,
        stream_path=payload.stream_path,
        recording_enabled=payload.recording_enabled,
        org_id=org_id,
        enabled=True,
    )
    db.add(cam)
    try:
        db.commit()
        db.refresh(cam)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, f"Camera '{payload.camera_id}' already exists.")

    stream_recorder_manager.sync_camera(cam.camera_id)

    actor_id, actor_name = _actor(user)
    audit_service.log_action(
        db, entity="camera", entity_id=cam.camera_id, entity_label=cam.name,
        action="create", summary=f'Created camera "{cam.name}" ({cam.camera_id}).',
        actor_id=actor_id, actor_name=actor_name,
    )

    return _to_camera_response(db, cam)


def update_camera(db: Session, user: dict, camera_id: str, payload: CameraUpdate) -> CameraResponse:
    org_id = user.get("org_id")
    cam = db.get(Camera, camera_id)
    if cam is None or cam.org_id != org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Camera '{camera_id}' not found.")

    update_data = payload.model_dump(
        exclude_unset=True, exclude={"zone_id", "priority_id", "stream_password", "status"},
    )
    for key, value in update_data.items():
        setattr(cam, key, value)

    if payload.zone_id is not None:
        cam.zone_id = _resolve_zone_or_422(db, org_id, payload.zone_id).id
    if payload.priority_id is not None:
        cam.priority_id = _resolve_priority_or_422(db, org_id, payload.priority_id).id
    if payload.stream_password:
        try:
            cam.stream_password_encrypted = encrypt_secret(payload.stream_password)
        except EmailCryptoNotConfigured as e:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))
    if payload.status is not None:
        cam.enabled = payload.status == "active"

    cam.updated_at = datetime.utcnow()
    db.add(cam)
    try:
        db.commit()
        db.refresh(cam)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Database constraint violation while updating camera.")

    stream_recorder_manager.sync_camera(camera_id)

    actor_id, actor_name = _actor(user)
    if payload.status is not None:
        action, verb = ("enable", "Activated") if payload.status == "active" else ("disable", "Deactivated")
        summary = f'{verb} camera "{cam.name}" ({cam.camera_id}).'
    else:
        action, summary = "update", f'Updated camera "{cam.name}" ({cam.camera_id}).'
    audit_service.log_action(
        db, entity="camera", entity_id=cam.camera_id, entity_label=cam.name,
        action=action, summary=summary, actor_id=actor_id, actor_name=actor_name,
    )

    return _to_camera_response(db, cam)


def delete_camera(db: Session, user: dict, camera_id: str) -> None:
    org_id = user.get("org_id")
    cam = db.get(Camera, camera_id)
    if cam is None or cam.org_id != org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Camera '{camera_id}' not found.")
    cam_name = cam.name

    db.delete(cam)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Database constraint violation while deleting camera.")

    stream_recorder_manager.sync_camera(camera_id)

    actor_id, actor_name = _actor(user)
    audit_service.log_action(
        db, entity="camera", entity_id=camera_id, entity_label=cam_name,
        action="delete", summary=f'Deleted camera "{cam_name}" ({camera_id}).',
        actor_id=actor_id, actor_name=actor_name,
    )
