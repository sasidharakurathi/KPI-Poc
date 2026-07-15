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
    org_id: Optional[int] = None           # FK to organizations.id (Phase 0)
    zone_id: Optional[int] = None          # FK to zones.id (Phase 1)
    priority_id: Optional[int] = None     # FK to priorities.id (Phase 1)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
