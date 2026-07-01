from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadResponse(BaseModel):
    job_id: str
    status: JobStatus
    filename: str
    created_at: datetime
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    kpis_requested: list[str] = []   # all KPI names the camera wants
    kpis_running: list[str] = []     # subset that are actually implemented
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    filename: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    kpis_running: list[str] = []
    kpi_results: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class KPIInfo(BaseModel):
    name: str
    display_name: str


class RegisteredKPIsResponse(BaseModel):
    count: int
    kpis: list[KPIInfo]


class CameraKPIDetail(BaseModel):
    kpi_id: int
    kpi_label: str
    implemented: bool


class CameraInfo(BaseModel):
    camera_id: str
    name: str
    zone: str
    priority: str
    kpi_ids: list[int]
    kpis: list[CameraKPIDetail]


class CameraListItem(BaseModel):
    camera_id: str
    name: str
    zone: str
    priority: str
    total_kpis: int
    implemented_kpis: int


class CameraListResponse(BaseModel):
    count: int
    cameras: list[CameraListItem]


class CameraCreate(BaseModel):
    camera_id: str
    name: str
    zone: str = ""
    priority: str = "medium"
    kpi_ids: list[int] = []


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    zone: Optional[str] = None
    priority: Optional[str] = None
    kpi_ids: Optional[list[int]] = None


class KPISettingsItem(BaseModel):
    name: str
    display_name: str
    enabled: bool
    config: dict[str, Any]


class KPISettingsResponse(BaseModel):
    count: int
    kpis: list[KPISettingsItem]
