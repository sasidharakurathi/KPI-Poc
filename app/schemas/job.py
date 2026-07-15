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
    kpis_requested: list[str] = []
    kpis_running: list[str] = []
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
