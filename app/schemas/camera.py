from typing import Optional

from pydantic import BaseModel


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
    stream_status: str = "disabled"
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
    stream_password: Optional[str] = None
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
    stream_password: Optional[str] = None
    stream_path: Optional[str] = None
    recording_enabled: Optional[bool] = None
