from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class EmailLogItem(BaseModel):
    model_config = {"from_attributes": True}

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
