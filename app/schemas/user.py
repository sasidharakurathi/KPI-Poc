"""User management schemas — Phase 7."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    employee_id: Optional[str] = None
    personal_email: EmailStr
    phone: Optional[str] = None
    address: Optional[str] = Field(default=None, max_length=250)
    username: str = Field(min_length=4, max_length=32, pattern=r"^[a-zA-Z0-9._-]+$")
    login_email: EmailStr
    role_id: int
    designation: Optional[str] = Field(default=None, max_length=60)
    description: Optional[str] = Field(default=None, max_length=250)


class UserUpdate(BaseModel):
    role_id: Optional[int] = None
    address: Optional[str] = Field(default=None, max_length=250)
    phone: Optional[str] = None
    designation: Optional[str] = Field(default=None, max_length=60)
    description: Optional[str] = Field(default=None, max_length=250)


class UserResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    employee_id: Optional[str] = None
    full_name: str
    designation: Optional[str] = None
    personal_email: Optional[str] = None
    phone: Optional[str] = None
    username: str
    login_email: str
    role_id: Optional[int] = None
    status: str
    created_at: datetime


class UserListResponse(BaseModel):
    count: int
    total: int
    users: list[UserResponse]
