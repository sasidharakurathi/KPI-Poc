"""Per-camera zone polygon lookup for KPIs that need a drawn zone
(BaseKPI.requires_zone = True) - e.g. a guard post or a counting area.

The polygon comes from KpiZoneLabel rows saved via
POST /api/cameras/{camera_id}/labels (app.services.kpi_label_service), not
config.json - config.json is deployment-wide, but a zone polygon is
inherently per-camera. KPIs only know their job_id, so this resolves
job_id -> Job.camera_id -> saved polygon for (camera_id, kpi_name).
"""
from typing import Optional


def get_camera_zone_points(job_id: str, kpi_name: str) -> Optional[list[list[float]]]:
    """None if no job/camera/saved polygon - callers should fall back to a
    sensible default (e.g. the full frame)."""
    if not job_id:
        return None

    from .. import db
    job = db.get_job(job_id)
    if job is None or not job.camera_id:
        return None

    from sqlmodel import select
    from ..db.engine import get_session_ctx
    from ..db.models.kpi_zone_label import KpiZoneLabel

    with get_session_ctx() as session:
        row = session.exec(
            select(KpiZoneLabel).where(
                KpiZoneLabel.camera_id == job.camera_id,
                KpiZoneLabel.kpi_name == kpi_name,
            )
        ).first()
        return list(row.points) if row and row.points else None
