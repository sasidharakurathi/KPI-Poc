from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.db import get_alert, query_alerts

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


def _parse_date(d: Optional[str]) -> Optional[datetime]:
    if not d:
        return None
    try:
        return datetime.fromisoformat(d)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid date '{d}' — expected ISO 8601.")


@router.get("")
async def list_alerts(
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
    rows, total = query_alerts(
        job_id=job_id, kpi_name=kpi_name, camera_id=camera_id, alert_type=alert_type,
        date_from=_parse_date(date_from), date_to=_parse_date(date_to),
        sort_by=sort_by, sort_dir=sort_dir, limit=limit, offset=offset,
    )
    return {"count": len(rows), "total": total, "alerts": rows}


@router.get("/{alert_id}")
async def get_alert_detail(alert_id: int):
    alert = get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    return alert


@router.get("/{alert_id}/frames/{position}")
async def get_alert_frame(alert_id: int, position: int):
    alert = get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    match = next((f for f in alert.get("frames", []) if f["position"] == position), None)
    if not match or not Path(match["path"]).exists():
        raise HTTPException(status_code=404, detail="Frame not found.")
    return FileResponse(path=match["path"], media_type="image/jpeg")


@router.get("/{alert_id}/labeled")
async def get_alert_labeled_frame(alert_id: int):
    alert = get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    match = next((f for f in alert.get("frames", []) if f.get("labeled_path")), None)
    if not match or not Path(match["labeled_path"]).exists():
        raise HTTPException(
            status_code=404,
            detail="Labeled frame not available (job not run in developer mode).",
        )
    return FileResponse(path=match["labeled_path"], media_type="image/jpeg")
