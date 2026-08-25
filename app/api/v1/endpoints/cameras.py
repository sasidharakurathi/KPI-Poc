"""Camera management endpoints - Phase 2.

Implements:
  GET    /api/cameras                List cameras for the caller's org, enriched
                                      with zone/priority name+color+level
  GET    /api/cameras/{camera_id}    Detail (full per-KPI implementation status)
  POST   /api/cameras                Create (validates zone_id/priority_id
                                      against the caller's org)
  PUT    /api/cameras/{camera_id}    Update (partial)
  DELETE /api/cameras/{camera_id}    Delete

org_id is always derived from the authenticated caller (require_permission),
never from the request body - same convention as every other Configuration
module. zone_id/priority_id are validated against the caller's own org.

kpi_ids stays the existing numeric list feeding the real detection pipeline
(app.kpis.registry) - the frontend's string-keyed kpi_model_ids belongs to
Phase 3's KPI Management capability table, which doesn't exist yet.

Business logic lives in app.services.camera_service.
"""
from fastapi import APIRouter, Depends

from app.core.dependencies import DbSession, require_permission
from app.schemas.camera import CamerasByZoneResponse, CameraCreate, CameraListResponse, CameraResponse, CameraUpdate
from app.services import camera_service

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("", response_model=CameraListResponse)
async def list_cameras(
    db: DbSession,
    user: dict = Depends(require_permission("cameras", "view")),
):
    return camera_service.list_cameras(db, user.get("org_id"))


@router.get("/by-zone", response_model=CamerasByZoneResponse)
async def list_cameras_by_zone(
    db: DbSession,
    user: dict = Depends(require_permission("cameras", "view")),
):
    """Every camera's camera_id + enabled status, grouped by zone.
    Registered before /{camera_id} so "by-zone" isn't swallowed as a
    camera_id path param (same convention as GET /api/config/zones/camera-counts)."""
    return camera_service.list_cameras_by_zone(db, user.get("org_id"))


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(
    camera_id: str,
    db: DbSession,
    user: dict = Depends(require_permission("cameras", "view")),
):
    return camera_service.get_camera(db, user.get("org_id"), camera_id)


@router.post("", response_model=CameraResponse, status_code=201)
async def create_camera(
    body: CameraCreate,
    db: DbSession,
    user: dict = Depends(require_permission("cameras", "create")),
):
    return camera_service.create_camera(db, user, body)


@router.put("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: str,
    body: CameraUpdate,
    db: DbSession,
    user: dict = Depends(require_permission("cameras", "edit")),
):
    return camera_service.update_camera(db, user, camera_id, body)


@router.delete("/{camera_id}", status_code=204)
async def delete_camera(
    camera_id: str,
    db: DbSession,
    user: dict = Depends(require_permission("cameras", "delete")),
):
    camera_service.delete_camera(db, user, camera_id)
