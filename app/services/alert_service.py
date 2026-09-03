"""Business logic for Phase 4: Alerts & Real-Time.

Kept out of app/api/v1/endpoints/alerts.py so the router stays a thin
request/response layer, matching the Phase 0/2/6/7 service-layer pattern.

Zone-scoping: a role with a non-empty zone_ids restriction (Phase 6) only
sees alerts whose camera belongs to one of those zones. Alerts with no
camera at all (job_id=None connectivity alerts with a deleted camera, or an
ad-hoc video upload never tied to a registered camera) are hidden from any
zone-restricted role - there's no zone to prove they're allowed to see it,
so the safe default is to hide rather than show. Unrestricted roles (empty
zone_ids) and the built-in Owner role see everything.

KPI-scoping: a role with a non-empty kpi_names restriction only sees alerts
for those KPIs (plus "system"/connectivity alerts, always visible to any
role that can see alerts at all) - see app.services.kpi_role_scope. Applied
as an AND alongside zone-scoping, same as zone-scoping is AND'd with the
org boundary.

A caller who can't see an alert gets a plain 404 (not 403) - same
enumeration-safety convention used everywhere else org/zone-scoping applies
in this app (e.g. Phase 2's camera lookups): the response is identical
whether the alert doesn't exist or just isn't visible to this caller.
"""
import csv
import io
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, status
from fastapi.responses import Response
from sqlmodel import Session

from app.db import get_alert as db_get_alert
from app.db import query_alerts as db_query_alerts
from app.schemas.alert import AlertResponse, AlertsResponse
from app.services.kpi_role_scope import allowed_kpi_names_for_user as _allowed_kpi_names
from app.services.zone_scope import allowed_camera_ids_for_user as _allowed_camera_ids


def list_alerts(
    db: Session, user: dict, *,
    job_id: Optional[str] = None, kpi_name: Optional[str] = None,
    camera_id: Optional[str] = None, alert_type: Optional[str] = None,
    date_from=None, date_to=None,
    sort_by: str = "created_at", sort_dir: str = "desc",
    limit: int = 100, offset: int = 0,
) -> AlertsResponse:
    allowed = _allowed_camera_ids(user, db)
    allowed_kpis = _allowed_kpi_names(user, db)
    rows, total = db_query_alerts(
        job_id=job_id, kpi_name=kpi_name, camera_id=camera_id, alert_type=alert_type,
        date_from=date_from, date_to=date_to, sort_by=sort_by, sort_dir=sort_dir,
        limit=limit, offset=offset, allowed_camera_ids=allowed, allowed_kpi_names=allowed_kpis,
        org_id=user.get("org_id"),
    )
    return AlertsResponse(count=len(rows), total=total, alerts=rows)


def _get_visible_alert_or_404(db: Session, user: dict, alert_id: int) -> dict:
    alert = db_get_alert(alert_id)
    if alert is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Alert '{alert_id}' not found.")

    if alert.get("org_id") != user.get("org_id"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Alert '{alert_id}' not found.")

    allowed = _allowed_camera_ids(user, db)
    if allowed is not None and alert.get("camera_id") not in allowed:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Alert '{alert_id}' not found.")

    allowed_kpis = _allowed_kpi_names(user, db)
    if allowed_kpis is not None and alert.get("kpi_name") not in allowed_kpis:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Alert '{alert_id}' not found.")
    return alert


def get_alert(db: Session, user: dict, alert_id: int) -> AlertResponse:
    return AlertResponse.model_validate(_get_visible_alert_or_404(db, user, alert_id))


def get_alert_frame_path(db: Session, user: dict, alert_id: int, position: int) -> str:
    alert = _get_visible_alert_or_404(db, user, alert_id)
    match = next((f for f in alert.get("frames", []) if f["position"] == position), None)
    if not match or not Path(match["path"]).exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Frame not found.")
    return match["path"]


def get_alert_labeled_frame_path(db: Session, user: dict, alert_id: int) -> str:
    alert = _get_visible_alert_or_404(db, user, alert_id)
    match = next((f for f in alert.get("frames", []) if f.get("labeled_path")), None)
    if not match or not Path(match["labeled_path"]).exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Labeled frame not available (job not run in developer mode).",
        )
    return match["labeled_path"]


_EXPORT_FIELDS = [
    ("Alert ID", "id"),
    ("Camera", "camera_name"),
    ("Camera ID", "camera_id"),
    ("KPI", "kpi_name"),
    ("Alert Type", "alert_type"),
    ("Confidence", "confidence"),
    ("Frame Index", "frame_idx"),
    ("Job ID", "job_id"),
    ("Created At", "created_at"),
]


def _export_csv(alert: dict) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Field", "Value"])
    for label, key in _EXPORT_FIELDS:
        writer.writerow([label, alert.get(key)])
    return buf.getvalue().encode("utf-8")


def _export_pdf(alert: dict) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas as pdf_canvas

    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, height - 1 * inch, f"Alert Report - #{alert.get('id')}")

    c.setFont("Helvetica", 11)
    y = height - 1.5 * inch
    for label, key in _EXPORT_FIELDS:
        c.drawString(1 * inch, y, f"{label}: {alert.get(key)}")
        y -= 0.3 * inch

    frames = alert.get("frames") or []
    image_path = next(
        (f.get("labeled_path") or f.get("path") for f in frames if (f.get("labeled_path") or f.get("path"))),
        None,
    )
    if image_path and Path(image_path).exists():
        try:
            y -= 0.2 * inch
            c.drawImage(image_path, 1 * inch, max(y - 3 * inch, 0.5 * inch), width=4 * inch, height=3 * inch, preserveAspectRatio=True)
        except Exception:# 
            pass

    c.showPage()
    c.save()
    return buf.getvalue()


def export_alert(db: Session, user: dict, alert_id: int, format: str) -> Response:
    if format not in ("pdf", "csv"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "format must be 'pdf' or 'csv'.")

    alert = _get_visible_alert_or_404(db, user, alert_id)

    alert = AlertResponse.model_validate(alert).model_dump(mode="json")

    if format == "csv":
        content = _export_csv(alert)
        media_type = "text/csv"
    else:
        content = _export_pdf(alert)
        media_type = "application/pdf"

    filename = f"alert_{alert_id}.{format}"
    return Response(
        content=content, media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
