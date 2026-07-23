"""Configuration-module schemas (Phase 1): priorities, zones, email servers, KPI models.

org_id is intentionally NOT accepted on any Create/Update schema — it's
derived server-side from the authenticated caller (see app/core/dependencies
.require_permission) and never trusted from client input. Letting a client
set/reassign org_id would let any caller create or move rows into a
different organization; see docs/IMPLEMENTATION_PLAN.md's Phase 1 audit
notes for the incident this fixes.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Priority ──────────────────────────────────────────────────────────────────

class PriorityCreate(BaseModel):
    name: str = Field(min_length=2, max_length=40)
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    level: int = Field(default=99, ge=1)  # severity rank, 1 = highest; default = unranked
    enabled: bool = True


class PriorityUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=40)
    color: Optional[str] = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    level: Optional[int] = Field(default=None, ge=1)
    enabled: Optional[bool] = None


class PriorityResponse(BaseModel):
    model_config = {"from_attributes": True}
    id: int
    name: str
    color: str
    level: int
    org_id: Optional[int] = None
    enabled: bool
    created_at: datetime


# ── Zone ──────────────────────────────────────────────────────────────────────

class ZoneCreate(BaseModel):
    name: str = Field(min_length=2, max_length=60)
    timezone_id: Optional[str] = None  # defaults to the organization's default_timezone_id if omitted
    description: Optional[str] = Field(default=None, max_length=200)
    enabled: bool = True


class ZoneUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=60)
    timezone_id: Optional[str] = None
    description: Optional[str] = Field(default=None, max_length=200)
    enabled: Optional[bool] = None


class ZoneResponse(BaseModel):
    model_config = {"from_attributes": True}
    id: int
    name: str
    timezone_id: Optional[str] = None
    description: Optional[str] = None
    org_id: Optional[int] = None
    enabled: bool
    created_at: datetime


# ── Email Server ──────────────────────────────────────────────────────────────

class EmailServerCreate(BaseModel):
    label: str = Field(min_length=2, max_length=60)
    smtp_host: str
    smtp_port: int = Field(ge=1, le=65535)
    username: str
    password: str                     # plaintext in transit; encrypted at rest
    use_tls: bool = True
    from_address: str
    from_name: str = Field(min_length=2, max_length=60)
    is_default: bool = False


class EmailServerUpdate(BaseModel):
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = Field(default=None, ge=1, le=65535)
    username: Optional[str] = None
    password: Optional[str] = None
    use_tls: Optional[bool] = None
    from_address: Optional[str] = None
    from_name: Optional[str] = Field(default=None, min_length=2, max_length=60)
    is_default: Optional[bool] = None
    enabled: Optional[bool] = None


class EmailServerResponse(BaseModel):
    model_config = {"from_attributes": True}
    id: int
    label: str
    smtp_host: str
    smtp_port: int
    username: str
    use_tls: bool
    from_address: str
    from_name: str
    enabled: bool
    is_default: bool
    org_id: Optional[int] = None
    password_set: bool = True       # never expose ciphertext


# ── KPI Model Catalog ─────────────────────────────────────────────────────────

class KpiModelCreate(BaseModel):
    name: str = Field(min_length=2, max_length=60)
    model_path: str
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    enabled: bool = True

class KpiModelResponse(BaseModel):
    """Shared across every organization — no org_id field, unlike the other
    Phase 1 config responses above (see app.db.models.domain_config
    .KpiModelCatalog's docstring)."""
    model_config = {"from_attributes": True}
    id: int
    name: str
    model_path: str
    confidence_threshold: float
    enabled: bool
    created_at: datetime
