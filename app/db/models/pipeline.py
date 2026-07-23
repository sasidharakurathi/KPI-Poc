from datetime import datetime
from typing import Optional

from sqlmodel import Column, Field, Relationship, SQLModel
from sqlalchemy import JSON as _JSON


class Job(SQLModel, table=True):
    __tablename__ = "jobs"  # type: ignore[assignment]

    job_id: str = Field(primary_key=True)
    filename: str
    video_path: str
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    status: str = Field(default="pending", index=True)
    kpis_running: list = Field(default_factory=list, sa_column=Column(_JSON))
    kpi_results: Optional[dict] = Field(default=None, sa_column=Column(_JSON))
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    alerts: list["Alert"] = Relationship(back_populates="job")


class Alert(SQLModel, table=True):
    __tablename__ = "alerts"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    # Nullable: Phase 4's camera-offline heartbeat monitor creates connectivity
    # alerts with no underlying video-processing job. camera_id is the direct,
    # always-populated link those alerts use (and is backfilled for existing
    # video-detection alerts too, from their job's camera_id) — see
    # app.db.create_alert and app.services.camera_heartbeat.
    job_id: Optional[str] = Field(default=None, foreign_key="jobs.job_id", index=True)
    camera_id: Optional[str] = Field(default=None, foreign_key="cameras.camera_id", index=True)
    # Resolved and stored at creation time from camera_id -> Camera.org_id
    # (see app.db.create_alert) so every alert query can filter by org
    # directly, without joining Camera — this is what actually enforces
    # "an org only ever sees its own alerts" (previously there was no
    # org-level filter at all for unrestricted/Owner-role callers, only
    # zone-scoping, which is a no-op when unrestricted).
    org_id: Optional[int] = Field(default=None, foreign_key="organizations.id", index=True)
    kpi_name: str
    alert_type: str
    frame_idx: Optional[int] = None  # no frame for a connectivity alert
    confidence: float = 1.0
    extra: Optional[dict] = Field(default=None, sa_column=Column(_JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    job: Optional[Job] = Relationship(back_populates="alerts")
    frames: list["AlertFrame"] = Relationship(back_populates="alert")


class AlertFrame(SQLModel, table=True):
    __tablename__ = "alert_frames"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    alert_id: int = Field(foreign_key="alerts.id", index=True)
    position: int
    frame_idx: int
    path: str
    labeled_path: Optional[str] = None

    alert: Optional[Alert] = Relationship(back_populates="frames")
