"""Business logic for zone-label drawing: a live frame to draw a polygon
on, and per-camera-per-KPI polygon storage for KPIs that need a zone
(BaseKPI.requires_zone) - occupancy_dwell, staff_absence, density_occupancy.

  GET  /api/cameras/{camera_id}/frame   -> get_camera_frame
  POST /api/cameras/{camera_id}/labels  -> save_camera_labels

The frame is grabbed live over the camera's own RTSP connection (same URL
builder streaming/recording already uses - app.stream_recorder.build_stream_url)
so the polygon lines up with the pixel geometry the real detection pipeline
sees. Cameras without a configured stream get a clear 422, not a silent
placeholder image - a polygon drawn against the wrong frame is worse than
no polygon.
"""
import base64
import logging
from datetime import datetime
from typing import Optional

import cv2
from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.config_loader import resolve_kpi_names
from app.db.models.camera import Camera
from app.db.models.kpi_zone_label import KpiZoneLabel
from app.kpis import get_registry
from app.schemas.kpi_zone_label import (
    CameraFrameResponse, SaveCameraLabelsRequest, SaveCameraLabelsResponse, SavedKpiZoneLabel,
)
from app.stream_recorder import build_stream_url

logger = logging.getLogger(__name__)

_OPEN_TIMEOUT_MSEC = 8_000
_READ_TIMEOUT_MSEC = 5_000


def _get_camera_or_404(db: Session, org_id: Optional[int], camera_id: str) -> Camera:
    cam = db.get(Camera, camera_id)
    if cam is None or cam.org_id != org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Camera '{camera_id}' not found.")
    return cam


def _assigned_kpi_names(cam: Camera) -> list[str]:
    """Same resolution order as videos.upload_video: legacy numeric
    kpi_ids first (real pipeline mapping), kpi_model_ids as fallback."""
    return resolve_kpi_names(cam.kpi_ids) or list(cam.kpi_model_ids)


def _grab_live_frame(cam: Camera):
    url = build_stream_url(cam)
    if not url:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Camera '{cam.camera_id}' has no camera_ip/RTSP stream configured - "
            "set up its stream (camera_ip, stream_path, credentials) before drawing labels.",
        )
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
