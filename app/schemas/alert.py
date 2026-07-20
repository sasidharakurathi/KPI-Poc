"""Alert schemas — Phase 4.

Field names/shape match the frontend's real contract exactly
(`vision-ai-frontend/src/types/api.ts` AlertRecord/AlertFrame/AlertsResponse)
— `src/api/alerts.ts` explicitly notes it already mirrors the real
GET /api/alerts query/response shapes, so no divergence to reconcile here
(unlike Auth, which had no pre-existing real contract to match).
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class AlertFrameResponse(BaseModel):
    model_config = {"from_attributes": True}
    id: int
    position: int
    frame_idx: int
    path: str
    labeled_path: Optional[str] = None


class AlertResponse(BaseModel):
    model_config = {"from_attributes": True}
    id: int
    job_id: Optional[str] = None
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    kpi_name: str
    alert_type: str
    frame_idx: Optional[int] = None
    confidence: float
    extra: Optional[dict] = None
    created_at: datetime
    frames: list[AlertFrameResponse] = []


class AlertsResponse(BaseModel):
    count: int
    total: int
    alerts: list[AlertResponse]
