"""Email Logs endpoint.

Implements:
  GET /api/email-logs   List (paginated, filterable) - org/zone/KPI-scoped,
                         same rule as GET /api/alerts (see app.services
                         .email_log_service) - an email log is the
                         notification record for an alert, so it's exactly
                         as visible as that alert is.
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import DbSession, require_permission
from app.schemas import EmailLogsResponse
from app.services import email_log_service

router = APIRouter(prefix="/api/email-logs", tags=["email-logs"])


def _parse_date(d: Optional[str]) -> Optional[datetime]:
    if not d:
        return None
    try:
        return datetime.fromisoformat(d)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid date '{d}' - expected ISO 8601.")


@router.get("", response_model=EmailLogsResponse)
async def get_email_logs(
    db: DbSession,
    user: dict = Depends(require_permission("alerts", "view")),
    status: Optional[str] = None,
    kpi_name: Optional[str] = None,
    camera_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort_by: str = "created_at",
    sort_dir: str = "desc",
    limit: int = 50,
    offset: int = 0,
):
    return email_log_service.list_email_logs(
        db, user, status=status, kpi_name=kpi_name, camera_id=camera_id,
        date_from=_parse_date(date_from), date_to=_parse_date(date_to),
        sort_by=sort_by, sort_dir=sort_dir, limit=limit, offset=offset,
    )
