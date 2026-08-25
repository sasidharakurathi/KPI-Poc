from datetime import datetime
from typing import Optional

from sqlmodel import Column, Field, SQLModel
from sqlalchemy import JSON as _JSON


class Camera(SQLModel, table=True):
    __tablename__ = "cameras"  

    camera_id: str = Field(primary_key=True)
    name: str
    zone: str = ""
    priority: str = "medium"
    kpi_ids: list = Field(default_factory=list, sa_column=Column(_JSON))

    
    camera_ip: Optional[str] = None
    rtsp_port: int = 554
    stream_username: Optional[str] = None
    stream_password_encrypted: Optional[str] = None
    stream_path: str = ""
    recording_enabled: bool = False

    
    camera_code: Optional[str] = None       
    mode: str = "live"                      
    connectivity_status: str = "pending"   
    enabled: bool = True                    
    org_id: Optional[int] = Field(default=None, foreign_key="organizations.id")
    zone_id: Optional[int] = Field(default=None, foreign_key="zones.id")
    priority_id: Optional[int] = Field(default=None, foreign_key="priorities.id")
    
    
    
    
    kpi_model_ids: list = Field(default_factory=list, sa_column=Column(_JSON))

    latitude: Optional[float] = None
    longitude: Optional[float] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
