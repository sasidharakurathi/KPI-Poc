"""Auth schemas — Phase 0.

Covers sign-up (org registration), sign-in, token refresh, and password reset.
"""
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class OrgRegisterRequest(BaseModel):
    company_name: str = Field(min_length=2, max_length=120)
    tagline: Optional[str] = Field(default=None, max_length=150)
    default_timezone: str = "UTC"
    site_name: str = Field(min_length=2, max_length=100)
    site_address: Optional[str] = None
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    owner_full_name: str = Field(min_length=2, max_length=100)
    owner_designation: Optional[str] = Field(default=None, max_length=60)
    owner_email: EmailStr
    owner_phone: Optional[str] = None
    username: str = Field(min_length=4, max_length=32, pattern=r"^[a-zA-Z0-9._-]+$")
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int   # seconds


class RefreshRequest(BaseModel):
    refresh_token: str


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(min_length=8, max_length=128)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8, max_length=128)
