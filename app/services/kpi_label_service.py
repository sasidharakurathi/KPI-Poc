"""Business logic for zone-label drawing: a live frame to draw a polygon
on, and per-camera-per-KPI polygon storage for KPIs that need a zone
(BaseKPI.requires_zone) - occupancy_dwell, staff_absence, density_occupancy.

  GET  /api/cameras/{camera_id}/frame    -> get_camera_frame
  GET  /api/cameras/{camera_id}/labels   -> list_camera_labels
  POST /api/cameras/{camera_id}/labels   -> save_camera_labels

The frame is grabbed live over the camera's own RTSP connection (same URL
builder streaming/recording already uses - app.stream_recorder.build_stream_url)
so the polygon lines up with the pixel geometry the real detection pipeline
sees. A camera that HAS a stream configured but is unreachable still gets a
clear error (503), not a silent placeholder - a polygon drawn against the
wrong frame is worse than no polygon. A camera with NO stream configured at
all falls back to a bundled sample frame (_SAMPLE_FRAME_PATH, committed to
the repo under app/assets/) so zone-drawing can still be exercised during
development/demos without a live camera; if that asset is ever missing, the
original 422 ("set up its stream...") is raised as before.
"""
import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.config_loader import resolve_kpi_names
from app.db.models.camera import Camera
from app.db.models.kpi_zone_label import KpiZoneLabel
from app.kpis import get_registry
from app.schemas.kpi_zone_label import (
    CameraFrameResponse, CameraKpiLabelInfo, CameraLabelsResponse,
    SaveCameraLabelsRequest, SaveCameraLabelsResponse, SavedKpiZoneLabel,
)
from app.stream_recorder import build_stream_url

logger = logging.getLogger(__name__)

_OPEN_TIMEOUT_MSEC = 8_000
_READ_TIMEOUT_MSEC = 5_000

# Dev/demo fallback: a single frame (extracted from a sample "people
# walking" clip, committed to the repo) stands in for a live frame when a
# camera has no camera_ip/RTSP stream configured yet, so zones can still be
# drawn without a live camera. See _grab_sample_frame.
_SAMPLE_FRAME_PATH = Path(__file__).resolve().parent.parent / "assets" / "sample_frames" / "people_walking.jpg"


def _get_camera_or_404(db: Session, org_id: Optional[int], camera_id: str) -> Camera:
    cam = db.get(Camera, camera_id)
    if cam is None or cam.org_id != org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Camera '{camera_id}' not found.")
    return cam


def _assigned_kpi_names(cam: Camera) -> list[str]:
    """Same resolution order as videos.upload_video: legacy numeric
    kpi_ids first (real pipeline mapping), kpi_model_ids as fallback."""
    return resolve_kpi_names(cam.kpi_ids) or list(cam.kpi_model_ids)


def _grab_sample_frame(cam: Camera):
    """Fallback for a camera with no RTSP stream configured - the bundled
    _SAMPLE_FRAME_PATH image instead of hard-blocking, so zone-drawing can
    still be exercised without a live camera. Raises the same 422 the caller
    would have gotten anyway if the asset is ever missing."""
    frame = cv2.imread(str(_SAMPLE_FRAME_PATH)) if _SAMPLE_FRAME_PATH.exists() else None
    if frame is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Camera '{cam.camera_id}' has no camera_ip/RTSP stream configured - "
            "set up its stream (camera_ip, stream_path, credentials) before drawing labels.",
        )
    return frame


def _grab_live_frame(cam: Camera):
    url = build_stream_url(cam)
    if not url:
        logger.info(
            f"[kpi_label] camera '{cam.camera_id}' has no RTSP stream configured - "
            "using sample fallback frame"
        )
        return _grab_sample_frame(cam)
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, _OPEN_TIMEOUT_MSEC)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, _READ_TIMEOUT_MSEC)
    try:
        if not cap.isOpened():
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                f"Could not connect to camera '{cam.camera_id}' stream.",
            )
        ok, frame = cap.read()
        if not ok or frame is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                f"Could not read a frame from camera '{cam.camera_id}' stream.",
            )
        return frame
    finally:
        cap.release()


def get_camera_frame(db: Session, org_id: Optional[int], camera_id: str) -> CameraFrameResponse:
    cam = _get_camera_or_404(db, org_id, camera_id)
    frame = _grab_live_frame(cam)
    h, w = frame.shape[:2]

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to encode frame.")
    frame_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return CameraFrameResponse(camera_id=camera_id, frame_base64=frame_b64, frame_width=w, frame_height=h)


def list_camera_labels(db: Session, org_id: Optional[int], camera_id: str) -> CameraLabelsResponse:
    """This camera's assigned KPIs, flagging which need a drawn zone
    (requires_zone) and returning any polygon already saved for them - no
    live camera connection needed, unlike get_camera_frame."""
    cam = _get_camera_or_404(db, org_id, camera_id)
    kpi_names = _assigned_kpi_names(cam)
    registry = get_registry()
    existing = {
        row.kpi_name: row
        for row in db.exec(select(KpiZoneLabel).where(KpiZoneLabel.camera_id == camera_id)).all()
    }

    kpis = [
        CameraKpiLabelInfo(
            kpi_name=name,
            display_name=registry[name].display_name if name in registry else name,
            requires_zone=bool(getattr(registry.get(name), "requires_zone", False)),
            points=existing[name].points if name in existing else None,
        )
        for name in kpi_names
    ]
    return CameraLabelsResponse(camera_id=camera_id, kpis=kpis)


def save_camera_labels(
    db: Session, org_id: Optional[int], user: dict, camera_id: str, payload: SaveCameraLabelsRequest,
) -> SaveCameraLabelsResponse:
    cam = _get_camera_or_404(db, org_id, camera_id)
    assigned = set(_assigned_kpi_names(cam))
    registry = get_registry()
    actor = user.get("username")

    saved: list[SavedKpiZoneLabel] = []
    for label in payload.labels:
        if label.kpi_name not in assigned:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"KPI '{label.kpi_name}' is not assigned to camera '{camera_id}'.",
            )
        cls = registry.get(label.kpi_name)
        if cls is None or not getattr(cls, "requires_zone", False):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"KPI '{label.kpi_name}' does not use a zone label.",
            )

        row = db.exec(
            select(KpiZoneLabel).where(
                KpiZoneLabel.camera_id == camera_id, KpiZoneLabel.kpi_name == label.kpi_name,
            )
        ).first()
        if row is None:
            row = KpiZoneLabel(camera_id=camera_id, kpi_name=label.kpi_name, created_by=actor)
        row.points = label.points
        row.updated_by = actor
        row.updated_at = datetime.utcnow()
        db.add(row)
        db.flush()
        saved.append(SavedKpiZoneLabel(kpi_name=row.kpi_name, points=row.points, updated_at=row.updated_at.isoformat()))

    db.commit()
    return SaveCameraLabelsResponse(camera_id=camera_id, labels=saved)
