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
    # JSON shape: Record<module, action[]> — see app.core.permissions
    # (PERMISSION_MODULES x PERMISSION_ACTIONS), matching the frontend's
    # PermissionMatrix type exactly, e.g. {"cameras": ["view", "edit"]}.
    default_email_server_id: Optional[int] = Field(default=None, foreign_key="email_servers.id")
    is_system: bool = Field(default=False)  # True for the built-in Owner role
    org_id: Optional[int] = Field(default=None, foreign_key="organizations.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None
