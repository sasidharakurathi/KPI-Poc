"""Zone-label drawing endpoints for KPIs that need a per-camera zone
polygon (BaseKPI.requires_zone) - e.g. occupancy_dwell, staff_absence,
density_occupancy.

  GET  /api/cameras/{camera_id}/frame    A live frame (base64) from this
       camera, to draw a zone polygon on.
  POST /api/cameras/{camera_id}/labels   Save/update one or more KPIs'
       zone polygons for this camera.

Business logic lives in app.services.kpi_label_service. Gated behind the
same "cameras" permission as the rest of camera configuration - drawing a
detection zone is camera setup, not a separate module.
"""
from fastapi import APIRouter, Depends

from app.core.dependencies import DbSession, require_permission
from app.schemas.kpi_zone_label import CameraFrameResponse, SaveCameraLabelsRequest, SaveCameraLabelsResponse
from app.services import kpi_label_service

router = APIRouter(prefix="/api/cameras", tags=["kpi-labels"])


@router.get("/{camera_id}/frame", response_model=CameraFrameResponse)
async def get_camera_frame(
    camera_id: str,
    db: DbSession,
    user: dict = Depends(require_permission("cameras", "view")),
):
    return kpi_label_service.get_camera_frame(db, user.get("org_id"), camera_id)


@router.post("/{camera_id}/labels", response_model=SaveCameraLabelsResponse)
async def save_labels(
    camera_id: str,
    payload: SaveCameraLabelsRequest,
    db: DbSession,
    user: dict = Depends(require_permission("cameras", "edit")),
):
    return kpi_label_service.save_camera_labels(db, user.get("org_id"), user, camera_id, payload)
