from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Organization(SQLModel, table=True):
    __tablename__ = "organizations"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    org_id: str = Field(unique=True, index=True)   # human-readable slug, e.g. "visionai-port"
    name: str                                       # 2-120 chars
    logo_path: Optional[str] = None
    tagline: Optional[str] = None
    # Superseded by default_timezone_id below (real FK into the static
    # timezones catalog) — left in place, unused, rather than dropped, per
    # this project's additive-only migration policy. Never read or written
    # by any endpoint anymore.
    default_timezone: str = "UTC"
    default_timezone_id: Optional[int] = Field(default=None, foreign_key="timezones.id")
    site_name: Optional[str] = None
    site_address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
