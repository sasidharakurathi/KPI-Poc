from datetime import datetime
from typing import Optional

from sqlmodel import Column, Field, SQLModel
from sqlalchemy import JSON as _JSON


class Role(SQLModel, table=True):
    __tablename__ = "roles"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True)          # 2-60 chars, unique within org
    description: Optional[str] = None
    permissions: Optional[dict] = Field(default=None, sa_column=Column(_JSON))
    # JSON structure: {"Dashboard": {"view_graphs": True, "view_cameras": True}, ...}
    default_email_server_id: Optional[int] = Field(default=None)  # FK to email_servers.id
    is_system: bool = Field(default=False)  # True for the built-in Owner role
    org_id: Optional[int] = Field(default=None)  # FK to organizations.id
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None
