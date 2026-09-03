"""Dashboard schemas - Phase 5.

These match the PRD's original spec-literal draft for /api/dashboard/* - see
app/services/dashboard_service.py's module docstring for why there's no
real frontend contract to match here (unlike most other phases).
"""
from typing import Optional

from pydantic import BaseModel


class DashboardSummaryResponse(BaseModel):
    total_cameras: int
    active_cameras: int
    inactive_cameras: int
    pending_cameras: int
    total_zones: int
    active_kpis: int  # enabled rows in the org's KPI Management catalog (app.db.models.kpi_configuration.KPIConfiguration)
    total_alerts: int
    alerts_last_24h: int
    alerts_by_priority: dict[str, int]


class AlertChartPoint(BaseModel):
    date: str        # ISO date, YYYY-MM-DD
    kpi_name: str
    count: int


class AlertChartResponse(BaseModel):
    camera_id: str
    date_from: str    # ISO date
    date_to: str      # ISO date
    points: list[AlertChartPoint]


class DashboardCameraItem(BaseModel):
    camera_id: str
    name: str
    zone_name: Optional[str] = None
    priority_name: Optional[str] = None
    priority_color: Optional[str] = None
    status: str                 # "active" | "inactive" - admin enabled toggle
    connectivity_status: str    # "active" | "inactive" | "pending" - live heartbeat-tracked
    stream_status: str          # "disabled" | "starting" | "connected" | "reconnecting" | "stopped"


class DashboardCamerasResponse(BaseModel):
    count: int
    cameras: list[DashboardCameraItem]
