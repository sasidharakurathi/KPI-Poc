"""Dashboard aggregation endpoints - Phase 5.

Implements:
  GET /api/dashboard/summary        Alert counts, camera status summary
  GET /api/dashboard/alert-chart    Alert counts per KPI over a date range for a camera
  GET /api/dashboard/cameras        Camera list with live connectivity status

These match the PRD's original spec-literal draft rather than a real
frontend contract - the actual Dashboard page composes its entire view from
GET /api/organization, /api/cameras, and /api/alerts (all already built) and
has no caller for any of these three. Built anyway for PRD completeness,
following the same call made for Phase 4's alert export endpoint. Zone-scoped
exactly like Alerts (see app.services.zone_scope).

Business logic lives in app.services.dashboard_service.
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import DbSession, require_permission
from app.schemas.dashboard import AlertChartResponse, DashboardCamerasResponse, DashboardSummaryResponse
from app.services import dashboard_service

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


def _parse_date(d: Optional[str]) -> Optional[datetime]:
    if not d:
        return None
    try:
        return datetime.fromisoformat(d)
    except ValueError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Invalid date '{d}' - expected ISO 8601.")


@router.get("/summary", response_model=DashboardSummaryResponse)
async def summary(
    db: DbSession,
    user: dict = Depends(require_permission("dashboard", "view")),
):
    return dashboard_service.get_summary(db, user)


@router.get("/alert-chart", response_model=AlertChartResponse)
async def alert_chart(
    camera_id: str,
    db: DbSession,
    user: dict = Depends(require_permission("dashboard", "view")),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    return dashboard_service.get_alert_chart(db, user, camera_id, _parse_date(date_from), _parse_date(date_to))


@router.get("/cameras", response_model=DashboardCamerasResponse)
async def cameras(
    db: DbSession,
    user: dict = Depends(require_permission("dashboard", "view")),
):
    return dashboard_service.get_cameras(db, user)
