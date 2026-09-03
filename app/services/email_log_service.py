"""Business logic for the Email Logs (alert-notification history) list.

Scoping matches app.services.alert_service exactly - org boundary, then
zone-scoping via camera_id, then KPI-scoping - since an email log is really
just the notification record for an alert, and should be exactly as visible
as that alert is. A camera-less log (e.g. a failed send with no default
EmailServer configured, logged before any camera lookup) is hidden from any
zone-restricted role, same rule as camera-less alerts.
"""
from datetime import datetime
from typing import Optional

from sqlmodel import Session

from app.db import query_email_logs as db_query_email_logs
from app.schemas.email_log import EmailLogsResponse
from app.services.kpi_role_scope import allowed_kpi_names_for_user as _allowed_kpi_names
from app.services.zone_scope import allowed_camera_ids_for_user as _allowed_camera_ids


def list_email_logs(
    db: Session, user: dict, *,
    status: Optional[str] = None, kpi_name: Optional[str] = None,
    camera_id: Optional[str] = None,
    date_from: Optional[datetime] = None, date_to: Optional[datetime] = None,
    sort_by: str = "created_at", sort_dir: str = "desc",
    limit: int = 50, offset: int = 0,
) -> EmailLogsResponse:
    allowed_cameras = _allowed_camera_ids(user, db)
    allowed_kpis = _allowed_kpi_names(user, db)
    rows, total = db_query_email_logs(
        status=status, kpi_name=kpi_name, camera_id=camera_id,
        date_from=date_from, date_to=date_to, sort_by=sort_by, sort_dir=sort_dir,
        limit=limit, offset=offset, allowed_camera_ids=allowed_cameras,
        allowed_kpi_names=allowed_kpis, org_id=user.get("org_id"),
    )
    return EmailLogsResponse(count=len(rows), total=total, logs=rows)
