"""Business logic for Phase 5: Dashboard.

These three endpoints match the PRD's original spec-literal draft
(GET /summary, /alert-chart, /cameras) rather than a real frontend contract:
the actual Dashboard page (vision-ai-frontend/src/pages/Dashboard) composes
its entire view from GET /api/organization, /api/cameras, and /api/alerts —
all already built in earlier phases — and has no caller for any of these
three. Built anyway for PRD completeness, the same call made for Phase 4's
alert export endpoint (frontend already does that client-side too).

Zone-scoping matches Alerts/the websocket exactly (see app.services.zone_scope)
— a zone-restricted role only sees cameras/alerts within its own zones.
"""
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import func as _func
from sqlmodel import Session, select

from app.db.models import Alert, Camera, Priority, Zone
from app.db.models.domain_config import KpiModelCatalog
from app.db.models.kpi_configuration import KPIConfiguration
from app.schemas.dashboard import (
    AlertChartPoint, AlertChartResponse, DashboardCameraItem,
    DashboardCamerasResponse, DashboardSummaryResponse,
)
from app.services.zone_scope import allowed_camera_ids_for_user
from app.stream_recorder import stream_recorder_manager

DEFAULT_CHART_WINDOW_DAYS = 30


def _visible_cameras(db: Session, user: dict) -> list[Camera]:
    org_id = user.get("org_id")
    allowed = allowed_camera_ids_for_user(user, db)
    cameras = db.exec(select(Camera).where(Camera.org_id == org_id)).all()
    if allowed is None:
        return cameras
    return [c for c in cameras if c.camera_id in allowed]


def get_summary(db: Session, user: dict) -> DashboardSummaryResponse:
    org_id = user.get("org_id")
    cameras = _visible_cameras(db, user)

    status_counts: dict[str, int] = {}
    priority_name_by_camera: dict[str, str] = {}
    priority_ids = {c.priority_id for c in cameras if c.priority_id is not None}
    priorities_by_id = (
        {p.id: p.name for p in db.exec(select(Priority).where(Priority.id.in_(priority_ids))).all()}
        if priority_ids else {}
    )
    for c in cameras:
        status_counts[c.connectivity_status] = status_counts.get(c.connectivity_status, 0) + 1
        priority_name_by_camera[c.camera_id] = priorities_by_id.get(c.priority_id, "Unassigned")

    # org_id is the real tenant boundary — always applied, regardless of
    # zone-scoping (allowed=None for an unrestricted role must never mean
    # "see every organization's alerts", only "see every zone within my
    # own org"). allowed narrows further within that org when set.
    allowed = allowed_camera_ids_for_user(user, db)
    alerts_stmt = select(Alert).where(Alert.org_id == org_id)
    if allowed is not None:
        alerts_stmt = alerts_stmt.where(Alert.camera_id.in_(allowed))
    alerts = db.exec(alerts_stmt).all()

    now = datetime.utcnow()
    alerts_last_24h = sum(1 for a in alerts if a.created_at >= now - timedelta(hours=24))

    alerts_by_priority: dict[str, int] = {}
    for a in alerts:
        key = priority_name_by_camera.get(a.camera_id, "Unassigned") if a.camera_id else "Unassigned"
        alerts_by_priority[key] = alerts_by_priority.get(key, 0) + 1

    total_zones = len(db.exec(select(Zone).where(Zone.org_id == org_id)).all())

    # Global counts, shared across every organization on the deployment —
    # KpiModelCatalog and KPIConfiguration are no longer org-scoped tables
    # (see both models' docstrings), so these are deployment-wide facts, not
    # per-org ones.
    active_kpi_models = db.exec(
        select(_func.count()).select_from(KpiModelCatalog).where(
            KpiModelCatalog.enabled == True,  # noqa: E712
        )
    ).one()

    # Distinct from active_kpi_models above: this counts entries in the KPI
    # Management catalog ("which detection capabilities are turned on" —
    # fire_smoke/ppe/etc., seeded via scripts/seed_kpi_catalog.py), not the
    # KPI Models/detection-model file catalog. enable_status defaults to
    # True at the column level, but a never-created row (no catalog entry
    # yet for a given KPI) isn't counted as active — only rows that exist.
    active_kpis = db.exec(
        select(_func.count()).select_from(KPIConfiguration).where(
            KPIConfiguration.enable_status == True,  # noqa: E712
        )
    ).one()

    return DashboardSummaryResponse(
        total_cameras=len(cameras),
        active_cameras=status_counts.get("active", 0),
        inactive_cameras=status_counts.get("inactive", 0),
        pending_cameras=status_counts.get("pending", 0),
        total_zones=total_zones,
        active_kpi_models=active_kpi_models,
        active_kpis=active_kpis,
        total_alerts=len(alerts),
        alerts_last_24h=alerts_last_24h,
        alerts_by_priority=alerts_by_priority,
    )


def get_alert_chart(
    db: Session, user: dict, camera_id: str,
    date_from: Optional[datetime], date_to: Optional[datetime],
) -> AlertChartResponse:
    cam = db.get(Camera, camera_id)
    if cam is None or cam.org_id != user.get("org_id"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Camera '{camera_id}' not found.")

    allowed = allowed_camera_ids_for_user(user, db)
    if allowed is not None and camera_id not in allowed:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Camera '{camera_id}' not found.")

    to_dt = date_to or datetime.utcnow()
    from_dt = date_from or (to_dt - timedelta(days=DEFAULT_CHART_WINDOW_DAYS))

    alerts = db.exec(
        select(Alert).where(
            Alert.camera_id == camera_id, Alert.created_at >= from_dt, Alert.created_at <= to_dt,
        )
    ).all()

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for a in alerts:
        day = a.created_at.date().isoformat()
        counts[(day, a.kpi_name)] += 1

    points = [
        AlertChartPoint(date=day, kpi_name=kpi_name, count=count)
        for (day, kpi_name), count in sorted(counts.items())
    ]

    return AlertChartResponse(
        camera_id=camera_id,
        date_from=from_dt.date().isoformat(),
        date_to=to_dt.date().isoformat(),
        points=points,
    )


def get_cameras(db: Session, user: dict) -> DashboardCamerasResponse:
    cameras = sorted(_visible_cameras(db, user), key=lambda c: c.camera_id)

    zone_ids = {c.zone_id for c in cameras if c.zone_id is not None}
    priority_ids = {c.priority_id for c in cameras if c.priority_id is not None}
    zones_by_id = {z.id: z for z in db.exec(select(Zone).where(Zone.id.in_(zone_ids))).all()} if zone_ids else {}
    priorities_by_id = (
        {p.id: p for p in db.exec(select(Priority).where(Priority.id.in_(priority_ids))).all()}
        if priority_ids else {}
    )

    # One query for every visible camera's alerts, bucketed by (camera_id,
    # year) in Python — same day-bucketing approach as get_alert_chart
    # above, just grouped by year and across every camera instead of one.
    camera_ids = [c.camera_id for c in cameras]
    alerts_by_year_by_camera: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    if camera_ids:
        alerts = db.exec(select(Alert).where(Alert.camera_id.in_(camera_ids))).all()
        for a in alerts:
            year = str(a.created_at.year)
            alerts_by_year_by_camera[a.camera_id][year] += 1

    items = []
    for c in cameras:
        zone = zones_by_id.get(c.zone_id)
        priority = priorities_by_id.get(c.priority_id)
        stream_status = stream_recorder_manager.status_for(c.camera_id)["status"]
        items.append(DashboardCameraItem(
            camera_id=c.camera_id,
            name=c.name,
            zone_name=zone.name if zone else None,
            priority_name=priority.name if priority else None,
            priority_color=priority.color if priority else None,
            status="active" if c.enabled else "inactive",
            connectivity_status=c.connectivity_status,
            stream_status=stream_status,
            alerts_by_year=dict(alerts_by_year_by_camera.get(c.camera_id, {})),
        ))

    return DashboardCamerasResponse(count=len(items), cameras=items)
