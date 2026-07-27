from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.db import query_email_logs
from app.schemas import EmailLogsResponse

router = APIRouter(prefix="/api/email-logs", tags=["email-logs"])


@router.get("", response_model=EmailLogsResponse)
async def get_email_logs(
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
    def _parse(d: Optional[str]) -> Optional[datetime]:
        if not d:
            return None
        try:
            return datetime.fromisoformat(d)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid date '{d}' - expected ISO 8601.")

    rows, total = query_email_logs(
        status=status, kpi_name=kpi_name, camera_id=camera_id,
        date_from=_parse(date_from), date_to=_parse(date_to),
        sort_by=sort_by, sort_dir=sort_dir, limit=limit, offset=offset,
    )
    return EmailLogsResponse(count=len(rows), total=total, logs=rows)
