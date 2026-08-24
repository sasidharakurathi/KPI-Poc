"""Per-camera, per-KPI zone polygon used by zone-based detectors
(BaseKPI.requires_zone = True - occupancy_dwell, staff_absence,
density_occupancy) to know WHERE in that camera's frame to look, e.g. a
guard post or a counting area. One polygon per (camera, kpi) pair, drawn
via POST /api/cameras/{camera_id}/labels against the frame returned by
GET /api/cameras/{camera_id}/label-frame - see app.services.kpi_label_service.

Points are [x, y] pixel coordinates in the coordinate space of the frame
they were drawn against (the live RTSP frame at label time) - same
convention as Research/*_prototype/zone_drawer.py.
"""
from datetime import datetime
from typing import Optional

from sqlmodel import Column, Field, SQLModel, UniqueConstraint
from sqlalchemy import JSON as _JSON


class KpiZoneLabel(SQLModel, table=True):
    __tablename__ = "kpi_zone_labels"
    __table_args__ = (UniqueConstraint("camera_id", "kpi_name", name="uq_kpi_zone_labels_camera_kpi"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    camera_id: str = Field(foreign_key="cameras.camera_id", index=True)
    kpi_name: str = Field(index=True)
    points: list = Field(sa_column=Column(_JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None
