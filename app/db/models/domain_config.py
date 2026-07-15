"""Reference-data tables: Priority, Timezone, Zone, EmailServer, KpiModelCatalog.
These are the Phase 1 Configuration module tables from the PRD."""
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Priority(SQLModel, table=True):
    __tablename__ = "priorities"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True)           # 2-40 chars, unique within org
    color: str                               # hex color, e.g. "#FF0000"
    enabled: bool = Field(default=True)
    org_id: Optional[int] = Field(default=None, foreign_key="organizations.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None


class Timezone(SQLModel, table=True):
    __tablename__ = "timezones"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    abbreviation: str
    timezone_name: str
    utc_offset: Optional[str] = None
    gmt_offset: Optional[str] = None
    enabled: bool = Field(default=True)


class Zone(SQLModel, table=True):
    __tablename__ = "zones"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str                                   # 2-60 chars
    org_id: Optional[int] = Field(default=None, foreign_key="organizations.id")
    timezone: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None


class EmailServer(SQLModel, table=True):
    __tablename__ = "email_servers"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    label: str = Field(unique=True)             # 2-60 chars
    smtp_host: str
    smtp_port: int = 587
    username: str
    password_encrypted: str                     # AES-encrypted, never returned in plaintext
    use_tls: bool = Field(default=True)
    from_address: str
    from_name: str
    enabled: bool = Field(default=True)
    is_default: bool = Field(default=False)     # org's default outgoing server
    org_id: Optional[int] = Field(default=None, foreign_key="organizations.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None


class KpiModelCatalog(SQLModel, table=True):
    """Catalog of AI model files available to the org. A KPI can use one or more entries."""
    __tablename__ = "kpi_model_catalog"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True)              # 2-60 chars
    model_path: str                             # path or URI to model weights
    confidence_threshold: float = 0.5          # 0.00-1.00
    enabled: bool = Field(default=True)
    org_id: Optional[int] = Field(default=None, foreign_key="organizations.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None
