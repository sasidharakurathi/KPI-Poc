"""User management schemas — Phase 7.

Field names/shape match the frontend's real contract (`vision-ai-frontend/
src/types/domain.ts` AppUser, `src/api/users.ts` UserCreateInput/
UserUpdateInput) — not the PRD-literal draft this file originally had.
Notably:
  - ids are surfaced as strings (DB uses int PKs; app.services.user_service
    converts at the boundary).
  - a single `email` field — the backend's User model has separate
    personal_email/login_email; both are set to this same value here.
  - status is 'active' | 'inactive' | 'deleted', mapped from the backend's
    'active' | 'disabled' | 'soft_deleted' | 'pending_verification'.
  - Divergence from the PRD flagged deliberately: the PRD says full_name is
    "not editable after creation", but the frontend's actual UserUpdateInput
    includes full_name as an editable field. Matching the frontend since
    that's the concrete, testable contract.
"""
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.core.validation import (
    validate_non_blank, validate_password_strength, validate_phone, validate_username,
)


class UserCreateInput(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    username: str
    email: EmailStr
    phone: str
    role_id: str
    password: str = Field(min_length=8, max_length=128)

    @field_validator("full_name")
    @classmethod
    def _full_name_non_blank(cls, v: str) -> str:
        return validate_non_blank(v, "full_name")

    @field_validator("username")
    @classmethod
    def _username(cls, v: str) -> str:
        return validate_username(v)

    @field_validator("phone")
    @classmethod
    def _phone(cls, v: str) -> str:
        return validate_phone(v)

    @field_validator("password")
    @classmethod
    def _password(cls, v: str) -> str:
        return validate_password_strength(v)


class UserUpdateInput(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    phone: str
    role_id: str

    @field_validator("full_name")
    @classmethod
    def _full_name_non_blank(cls, v: str) -> str:
        return validate_non_blank(v, "full_name")

    @field_validator("phone")
    @classmethod
    def _phone(cls, v: str) -> str:
        return validate_phone(v)


class UserStatusUpdate(BaseModel):
    status: str  # "active" | "inactive"

    @field_validator("status")
    @classmethod
    def _valid_status(cls, v: str) -> str:
        if v not in ("active", "inactive"):
            raise ValueError('status must be "active" or "inactive".')
        return v


class UserPasswordReset(BaseModel):
    new_password: str = Field(min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def _password(cls, v: str) -> str:
        return validate_password_strength(v)


class UserResponse(BaseModel):
    id: str
    full_name: str
    username: str
    email: str
    phone: str
    role_id: Optional[str] = None
    status: str
    must_change_password: bool
    mfa_enabled: bool
    last_login_at: Optional[str] = None
    created_at: str
