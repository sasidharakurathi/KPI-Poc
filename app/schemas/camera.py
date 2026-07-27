"""Camera schemas - Phase 2.

Field names/shape match the frontend's real contract (`vision-ai-frontend/
src/types/domain.ts` CameraRecord/CameraCreateInput/CameraUpdateInput,
`src/api/cameras.ts` CameraListRow's enrichment fields) where a real contract
exists. Notably:
  - zone_id/priority_id are string-typed at the API boundary (DB uses int
    FKs into zones/priorities; app.services.camera_service converts at the
    boundary), matching the project's established id convention.
  - status is "active" | "inactive", mapped from the backend's own
    Camera.enabled boolean toggle - distinct from Camera.connectivity_status,
    which is a live health signal, not something a client sets directly.
  - kpi_ids stays the existing numeric list (feeds the real detection
    pipeline via app.kpis.registry) - kpi_model_ids (string-keyed, Phase 3's
    KPI Management capability keys) is a separate, additive field validated
    against the real registered-detector list.
  - zone_name/priority_name/priority_color/priority_level are server-side
    enrichment (the frontend's mock does this client-side today via
    CameraListRow) so a future real-API swap needs no client-side join logic.
"""
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.core.validation import validate_non_blank


class CameraKPIDetail(BaseModel):
    kpi_id: int
    kpi_label: str
    implemented: bool


class CameraCreate(BaseModel):
    camera_id: str = Field(min_length=2, max_length=50)
    name: str = Field(min_length=2, max_length=100)
    zone_id: str
    priority_id: str
    kpi_ids: list[int] = Field(default_factory=list)
    kpi_model_ids: list[str] = Field(default_factory=list)
    camera_ip: Optional[str] = None
    rtsp_port: int = Field(default=554, ge=1, le=65535)
    stream_username: Optional[str] = None
    stream_password: Optional[str] = None
    stream_path: str = ""
    recording_enabled: bool = False

    @field_validator("camera_id")
    @classmethod
    def _camera_id_non_blank(cls, v: str) -> str:
        return validate_non_blank(v, "camera_id")

    @field_validator("name")
    @classmethod
    def _name_non_blank(cls, v: str) -> str:
        return validate_non_blank(v, "name")


class CameraUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=100)
    zone_id: Optional[str] = None
    priority_id: Optional[str] = None
    kpi_ids: Optional[list[int]] = None
    kpi_model_ids: Optional[list[str]] = None
    camera_ip: Optional[str] = None
    rtsp_port: Optional[int] = Field(default=None, ge=1, le=65535)
    stream_username: Optional[str] = None
    stream_password: Optional[str] = None
    stream_path: Optional[str] = None
    recording_enabled: Optional[bool] = None
    status: Optional[str] = None  # "active" | "inactive"

    @field_validator("name")
    @classmethod
    def _name_non_blank(cls, v: str) -> str:
        return validate_non_blank(v, "name") if v is not None else v

    @field_validator("status")
    @classmethod
    def _valid_status(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("active", "inactive"):
            raise ValueError('status must be "active" or "inactive".')
        return v


class CameraResponse(BaseModel):
    camera_id: str
    name: str
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    priority_id: Optional[str] = None
    priority_name: Optional[str] = None
    priority_color: Optional[str] = None
    priority_level: Optional[int] = None
    kpi_ids: list[int]
    kpis: list[CameraKPIDetail]
    kpi_model_ids: list[str] = Field(default_factory=list)
    kpi_model_labels: list[str] = Field(default_factory=list)
    status: str
    camera_ip: Optional[str] = None
    rtsp_port: int = 554
    stream_username: Optional[str] = None
    stream_path: str = ""
    stream_password_set: bool = False
    recording_enabled: bool = False
    stream_status: str = "disabled"
    stream_error: Optional[str] = None
    created_at: str
    alerts_by_year: dict[str, int] = Field(default_factory=dict)  # {"2025": 12, "2026": 45, ...} - this camera's own alerts, every KPI combined


class CameraListItem(BaseModel):
    camera_id: str
    name: str
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    priority_id: Optional[str] = None
    priority_name: Optional[str] = None
    priority_color: Optional[str] = None
    priority_level: Optional[int] = None
    total_kpis: int
    implemented_kpis: int
    kpi_model_ids: list[str] = Field(default_factory=list)
    kpi_model_labels: list[str] = Field(default_factory=list)
    status: str
    recording_enabled: bool = False
    stream_status: str = "disabled"
    created_at: str


class CameraListResponse(BaseModel):
    count: int
    cameras: list[CameraListItem]
