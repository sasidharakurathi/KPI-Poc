from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel
from sqlalchemy import Column, BigInteger, Integer, String, Boolean, TIMESTAMP, text
from sqlalchemy import JSON as _JSON

class KPIConfiguration(SQLModel, table=True):
    """Phase 3 (KPI Management). kpi_name is gated to real registered
    detectors (app.kpis.registry) — see app.api.v1.endpoints.kpis. Writing
    enable_status/parameters now also writes through to config.json (see
    app.config_loader.update_kpi_config), which is what the real detection
    pipeline actually reads; assigned_models does not write through (no
    sensible 1:1 mapping to config.json's single model_path per KPI)."""
    __tablename__ = "kpi_configuration"

    # BigInteger with an explicit variant for SQLite: SQLite only grants its
    # built-in autoincrementing-rowid behavior to a column declared as
    # exactly "INTEGER PRIMARY KEY" — a BigInteger PK (as originally written)
    # silently loses autoincrement there, though it works fine on Postgres
    # (a real BIGINT + sequence). Only matters for the pytest SQLite DB.
    id: Optional[int] = Field(
        default=None,
        sa_column=Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True),
    )
    kpi_name: str = Field(sa_column=Column(String(255), nullable=False))
    org_id: Optional[int] = Field(default=None, foreign_key="organizations.id")
    description: Optional[str] = Field(default=None, sa_column=Column(String(500)))
    category: Optional[str] = Field(default=None, sa_column=Column(String(50)))
    assigned_models: Optional[list] = Field(default=None, sa_column=Column(_JSON))
    parameters: Optional[dict] = Field(default=None, sa_column=Column(_JSON))
    enable_status: Optional[bool] = Field(default=True, sa_column=Column(Boolean, server_default=text("TRUE")))
    created_at: Optional[datetime] = Field(default=None, sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP")))
    created_by: Optional[str] = Field(default=None, sa_column=Column(String(100)))
    updated_at: Optional[datetime] = Field(default=None, sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP")))
    updated_by: Optional[str] = Field(default=None, sa_column=Column(String(100)))
