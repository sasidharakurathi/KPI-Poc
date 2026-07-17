from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    __tablename__ = "users"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    employee_id: Optional[str] = None
    full_name: str                                # 2-100 chars; not editable after creation
    designation: Optional[str] = None
    personal_email: Optional[str] = Field(default=None, unique=True)
    phone: Optional[str] = None                  # E.164 format if provided
    address: Optional[str] = None
    username: str = Field(unique=True, index=True)  # 4-32 chars
    login_email: str = Field(unique=True)
    password_hash: str
    org_id: Optional[int] = Field(default=None, foreign_key="organizations.id")
    role_id: Optional[int] = Field(default=None, foreign_key="roles.id")
    status: str = Field(default="active")
    # "pending_verification" | "active" | "disabled" | "soft_deleted"
    verify_status: bool = Field(default=False)
    force_password_change: bool = Field(default=False)
    failed_login_attempts: int = Field(default=0)
    locked_until: Optional[datetime] = None
    # Reused for both the Phase 0 activation link and the forgot-password flow
    # (single-use, time-limited — only one flow is ever "in flight" per account).
    reset_token: Optional[str] = None
    reset_token_expires: Optional[datetime] = None
    last_login_at: Optional[datetime] = None
    mfa_enabled: bool = Field(default=False)  # PRD: optional TOTP MFA, off by default
    # Bumped by auth_service.revoke_all_sessions() (logout, password reset, and
    # future disable/delete). Embedded in every access JWT at issuance; a
    # mismatch on a later request means that token was minted before the last
    # revocation and is rejected even though it hasn't expired yet.
    token_version: int = Field(default=0)
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None


class RefreshToken(SQLModel, table=True):
    """Server-tracked refresh tokens for JWT session management."""
    __tablename__ = "refresh_tokens"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)                # FK to users.id
    token_hash: str = Field(unique=True)            # SHA-256 of the raw token
    expires_at: datetime
    revoked: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
