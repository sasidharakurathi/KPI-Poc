from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Organization(SQLModel, table=True):
    __tablename__ = "organizations"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    org_id: str = Field(unique=True, index=True)   # human-readable slug, e.g. "jana-port"
    name: str                                       # 2-120 chars
    logo_path: Optional[str] = None
    tagline: Optional[str] = None
    default_timezone: str = "UTC"
    site_name: Optional[str] = None
    site_address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
