from datetime import datetime
from typing import Optional

from sqlmodel import Column, Field, SQLModel
from sqlalchemy import JSON as _JSON


class EmailLog(SQLModel, table=True):
    """One row per attempted alert-notification email (sent or failed)."""
    __tablename__ = "email_logs"  

    id: Optional[int] = Field(default=None, primary_key=True)
    alert_id: Optional[int] = Field(default=None, index=True)
    org_id: Optional[int] = Field(default=None, foreign_key="organizations.id", index=True)
    kpi_name: Optional[str] = Field(default=None, index=True)
    alert_type: Optional[str] = None
    camera_id: Optional[str] = Field(default=None, index=True)
    camera_name: Optional[str] = None
    subject: str = ""
    recipients: list = Field(default_factory=list, sa_column=Column(_JSON))
    status: str = Field(default="sent", index=True)   
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
