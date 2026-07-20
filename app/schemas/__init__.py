"""Backward-compatible re-export: existing code imports from app.schemas directly."""
from .auth import (
    ActivateRequest, ChangePasswordRequest, LoginRequest, LoginResponse,
    MeResponse, OrgRegisterRequest, PasswordResetConfirm, PasswordResetRequest,
    RefreshRequest, RegisterResponse, TokenResponse,
)
from .camera import (
    CameraCreate, CameraKPIDetail, CameraListItem,
    CameraListResponse, CameraResponse, CameraUpdate,
)
from .config import (
    EmailServerCreate, EmailServerResponse, EmailServerUpdate,
    KpiModelCreate, KpiModelResponse,
    PriorityCreate, PriorityResponse,
    ZoneCreate, ZoneResponse,
)
from .email_log import EmailLogItem, EmailLogsResponse
from .job import JobStatus, JobStatusResponse, UploadResponse
from .kpi import KPIInfo, KPISettingsItem, KPISettingsResponse, RegisteredKPIsResponse
from .organization import OrganizationResponse, OrganizationUpdate
from .role import RoleInput, RoleResponse
from .settings import (
    EmailSettingsResponse, EmailSettingsUpdate,
    TimezoneSettingsResponse, TimezoneSettingsUpdate,
)
from .user import UserCreateInput, UserPasswordReset, UserResponse, UserStatusUpdate, UserUpdateInput

__all__ = [
    "ActivateRequest", "ChangePasswordRequest", "LoginRequest", "LoginResponse",
    "MeResponse", "OrgRegisterRequest", "PasswordResetConfirm", "PasswordResetRequest",
    "RefreshRequest", "RegisterResponse", "TokenResponse",
    "CameraCreate", "CameraKPIDetail", "CameraListItem",
    "CameraListResponse", "CameraResponse", "CameraUpdate",
    "EmailServerCreate", "EmailServerResponse", "EmailServerUpdate",
    "KpiModelCreate", "KpiModelResponse",
    "PriorityCreate", "PriorityResponse",
    "ZoneCreate", "ZoneResponse",
    "EmailLogItem", "EmailLogsResponse",
    "JobStatus", "JobStatusResponse", "UploadResponse",
    "KPIInfo", "KPISettingsItem", "KPISettingsResponse", "RegisteredKPIsResponse",
    "OrganizationResponse", "OrganizationUpdate",
    "RoleInput", "RoleResponse",
    "EmailSettingsResponse", "EmailSettingsUpdate",
    "TimezoneSettingsResponse", "TimezoneSettingsUpdate",
    "UserCreateInput", "UserPasswordReset", "UserResponse", "UserStatusUpdate", "UserUpdateInput",
]
