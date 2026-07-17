"""Auth schemas — Phase 0.

Covers org registration/activation, sign-in, token refresh, "me", and
password reset/change. Field names follow the PRD's own field tables
(§2.1-2.4) since auth has no pre-existing real frontend contract to match
(see docs/IMPLEMENTATION_PLAN.md, "Critical contract-mapping notes").
"""
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

from app.core.validation import validate_non_blank, validate_password_strength as _validate_password_strength


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
    confirm_password: str

    @field_validator("company_name")
    @classmethod
    def _company_name_non_blank(cls, v: str) -> str:
        return validate_non_blank(v, "company_name")

    @field_validator("site_name")
    @classmethod
    def _site_name_non_blank(cls, v: str) -> str:
        return validate_non_blank(v, "site_name")

    @field_validator("owner_full_name")
    @classmethod
    def _owner_full_name_non_blank(cls, v: str) -> str:
        return validate_non_blank(v, "owner_full_name")

    @field_validator("password")
    @classmethod
    def _password_strength(cls, v: str) -> str:
        return _validate_password_strength(v)

    @model_validator(mode="after")
    def _passwords_match(self) -> "OrgRegisterRequest":
        if self.password != self.confirm_password:
            raise ValueError("password and confirm_password do not match.")
        return self


class RegisterResponse(BaseModel):
    organization_name: str
    username: str
    message: str


class ActivateRequest(BaseModel):
    token: str


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user_id: int
    username: str
    full_name: str
    role_id: int
    role_name: str
    org_id: Optional[int] = None
    force_password_change: bool = False


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str


class MeResponse(BaseModel):
    id: int
    username: str
    full_name: str
    login_email: str
    personal_email: Optional[str] = None
    role_id: Optional[int] = None
    role_name: Optional[str] = None
    org_id: Optional[int] = None
    permissions: dict = Field(default_factory=dict)
    force_password_change: bool = False
    status: str


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def _password_strength(cls, v: str) -> str:
        return _validate_password_strength(v)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def _password_strength(cls, v: str) -> str:
        return _validate_password_strength(v)
