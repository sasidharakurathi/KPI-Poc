"""Role & permissions schemas — Phase 6."""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class RoleCreate(BaseModel):
    name: str = Field(min_length=2, max_length=60)
    description: Optional[str] = Field(default=None, max_length=250)
    permissions: Optional[dict[str, Any]] = None
    default_email_server_id: Optional[int] = None


class RoleUpdate(BaseModel):
    description: Optional[str] = Field(default=None, max_length=250)
    permissions: Optional[dict[str, Any]] = None
    default_email_server_id: Optional[int] = None


class RoleResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    name: str
    description: Optional[str] = None
    permissions: Optional[dict[str, Any]] = None
    is_system: bool
    created_at: datetime


class RoleListResponse(BaseModel):
    count: int
    roles: list[RoleResponse]
