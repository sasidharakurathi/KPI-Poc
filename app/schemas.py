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
    camera_ip: Optional[str] = None
    rtsp_port: int = 554
    stream_username: Optional[str] = None
    stream_path: str = ""
    stream_password_set: bool = False
    recording_enabled: bool = False
    stream_status: str = "disabled"     # disabled | starting | connected | reconnecting | stopped
    stream_error: Optional[str] = None


class CameraListItem(BaseModel):
    camera_id: str
    name: str
    zone: str
    priority: str
    total_kpis: int
    implemented_kpis: int
    recording_enabled: bool = False
    stream_status: str = "disabled"


class CameraListResponse(BaseModel):
    count: int
    cameras: list[CameraListItem]


class CameraCreate(BaseModel):
    camera_id: str
    name: str
    zone: str = ""
    priority: str = "medium"
    kpi_ids: list[int] = []
    camera_ip: Optional[str] = None
    rtsp_port: int = 554
    stream_username: Optional[str] = None
    stream_password: Optional[str] = None   # plaintext in transit; encrypted before storage, never echoed back
    stream_path: str = ""
    recording_enabled: bool = False


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    zone: Optional[str] = None
    priority: Optional[str] = None
    kpi_ids: Optional[list[int]] = None
    camera_ip: Optional[str] = None
    rtsp_port: Optional[int] = None
    stream_username: Optional[str] = None
    stream_password: Optional[str] = None   # blank/omitted = keep the existing stored password
    stream_path: Optional[str] = None
    recording_enabled: Optional[bool] = None


class KPISettingsItem(BaseModel):
    name: str
    display_name: str
    enabled: bool
    config: dict[str, Any]


class KPISettingsResponse(BaseModel):
    count: int
    kpis: list[KPISettingsItem]


class EmailSettingsResponse(BaseModel):
    enabled: bool
    smtp_host: str
    smtp_port: int
    smtp_username: str
    use_tls: bool
    from_address: str
    from_name: str
    recipients: list[str]
    password_set: bool          # never expose the password/ciphertext itself
    updated_at: Optional[datetime] = None


class EmailSettingsUpdate(BaseModel):
    enabled: Optional[bool] = None
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None   # plaintext in transit; encrypted before storage, never echoed back
    use_tls: Optional[bool] = None
    from_address: Optional[str] = None
    from_name: Optional[str] = None
    recipients: Optional[list[str]] = None


class TimezoneSettingsResponse(BaseModel):
    default: str
    updated_at: Optional[datetime] = None


class TimezoneSettingsUpdate(BaseModel):
    default: str


class EmailLogItem(BaseModel):
    model_config = {"from_attributes": True}   # allows building directly from db.EmailLog ORM rows

    id: int
    alert_id: Optional[int] = None
    kpi_name: Optional[str] = None
    alert_type: Optional[str] = None
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    subject: str
    recipients: list[str]
    status: str
    error: Optional[str] = None
    created_at: datetime


class EmailLogsResponse(BaseModel):
    count: int
    total: int
    logs: list[EmailLogItem]
