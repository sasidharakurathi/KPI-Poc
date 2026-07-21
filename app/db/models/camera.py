from datetime import datetime
from typing import Optional

from sqlmodel import Column, Field, SQLModel
from sqlalchemy import JSON as _JSON


class Camera(SQLModel, table=True):
    __tablename__ = "cameras"  # type: ignore[assignment]

    camera_id: str = Field(primary_key=True)
    name: str
    zone: str = ""
    priority: str = "medium"
    kpi_ids: list = Field(default_factory=list, sa_column=Column(_JSON))

    # stream credentials (password stored encrypted)
    camera_ip: Optional[str] = None
    rtsp_port: int = 554
    stream_username: Optional[str] = None
    stream_password_encrypted: Optional[str] = None
    stream_path: str = ""
    recording_enabled: bool = False

    # PRD extensions (added via migration for existing DBs)
    camera_code: Optional[str] = None       # human-facing ID per PRD
    mode: str = "live"                      # "live" | "2min_clips"
    connectivity_status: str = "pending"   # "active" | "inactive" | "pending"
    enabled: bool = True                    # admin enable/disable toggle
    org_id: Optional[int] = Field(default=None, foreign_key="organizations.id")
    zone_id: Optional[int] = Field(default=None, foreign_key="zones.id")
    priority_id: Optional[int] = Field(default=None, foreign_key="priorities.id")
    # String-keyed KPI Management capability keys (Phase 3's KPIConfiguration.kpi_name,
    # gated to real registered detectors) — distinct from the legacy numeric
    # kpi_ids above, which is what the pipeline actually reads. Matches the
    # frontend's CameraRecord.kpi_model_ids.
    kpi_model_ids: list = Field(default_factory=list, sa_column=Column(_JSON))

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
