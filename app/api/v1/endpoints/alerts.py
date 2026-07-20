"""Alert endpoints — Phase 4.

Implements:
  GET  /api/alerts                        List (paginated, filterable) — zone-scoped
  GET  /api/alerts/{alert_id}              Detail
  GET  /api/alerts/{alert_id}/frames/{position}   Raw frame image
  GET  /api/alerts/{alert_id}/labeled      Labeled frame image (developer mode only)
  GET  /api/alerts/{alert_id}/export       ?format=pdf|csv — download a report

Zone-scoping: a role with a non-empty zone_ids restriction (Phase 6) only
sees alerts from cameras in those zones — see app.services.alert_service for
the exact rule (camera-less alerts are hidden from restricted roles).

For real-time updates see GET (WebSocket) /api/ws/alerts in ws.py — this
router only covers the request/response REST surface.

Business logic lives in app.services.alert_service.
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from app.core.dependencies import DbSession, require_permission
from app.schemas.alert import AlertResponse, AlertsResponse
from app.services import alert_service

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


def _parse_date(d: Optional[str]) -> Optional[datetime]:
    if not d:
        return None
    try:
        return datetime.fromisoformat(d)
    except ValueError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Invalid date '{d}' — expected ISO 8601.")


@router.get("", response_model=AlertsResponse)
async def list_alerts(
    db: DbSession,
    user: dict = Depends(require_permission("alerts", "view")),
    job_id: Optional[str] = None,
    kpi_name: Optional[str] = None,
    camera_id: Optional[str] = None,
    alert_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort_by: str = "created_at",
    sort_dir: str = "desc",
    limit: int = 100,
    offset: int = 0,
):
    return alert_service.list_alerts(
        db, user, job_id=job_id, kpi_name=kpi_name, camera_id=camera_id, alert_type=alert_type,
        date_from=_parse_date(date_from), date_to=_parse_date(date_to),
        sort_by=sort_by, sort_dir=sort_dir, limit=limit, offset=offset,
    )


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert_detail(
    alert_id: int,
    db: DbSession,
    user: dict = Depends(require_permission("alerts", "view")),
):
    return alert_service.get_alert(db, user, alert_id)


@router.get("/{alert_id}/frames/{position}")
async def get_alert_frame(
    alert_id: int,
    position: int,
    db: DbSession,
    user: dict = Depends(require_permission("alerts", "view")),
):
    path = alert_service.get_alert_frame_path(db, user, alert_id, position)
    return FileResponse(path=path, media_type="image/jpeg")


@router.get("/{alert_id}/labeled")
async def get_alert_labeled_frame(
    alert_id: int,
    db: DbSession,
    user: dict = Depends(require_permission("alerts", "view")),
):
    path = alert_service.get_alert_labeled_frame_path(db, user, alert_id)
    return FileResponse(path=path, media_type="image/jpeg")


@router.get("/{alert_id}/export")
async def export_alert(
    alert_id: int,
    db: DbSession,
    user: dict = Depends(require_permission("alerts", "view")),
    format: str = "pdf",
):
    return alert_service.export_alert(db, user, alert_id, format)
