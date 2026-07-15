"""Backward-compatible re-export: existing code imports from app.schemas directly."""
from .auth import (
    ActivateRequest, ChangePasswordRequest, LoginRequest, LoginResponse,
    MeResponse, OrgRegisterRequest, PasswordResetConfirm, PasswordResetRequest,
    RefreshRequest, RegisterResponse, TokenResponse,
)
from .camera import (
    CameraCreate, CameraInfo, CameraKPIDetail,
    CameraListItem, CameraListResponse, CameraUpdate,
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
from .role import RoleCreate, RoleListResponse, RoleResponse, RoleUpdate
from .settings import (
    EmailSettingsResponse, EmailSettingsUpdate,
    TimezoneSettingsResponse, TimezoneSettingsUpdate,
)
from .user import UserCreate, UserListResponse, UserResponse, UserUpdate

__all__ = [
    "ActivateRequest", "ChangePasswordRequest", "LoginRequest", "LoginResponse",
    "MeResponse", "OrgRegisterRequest", "PasswordResetConfirm", "PasswordResetRequest",
    "RefreshRequest", "RegisterResponse", "TokenResponse",
    "CameraCreate", "CameraInfo", "CameraKPIDetail",
    "CameraListItem", "CameraListResponse", "CameraUpdate",
    "EmailServerCreate", "EmailServerResponse", "EmailServerUpdate",
    "KpiModelCreate", "KpiModelResponse",
    "PriorityCreate", "PriorityResponse",
    "ZoneCreate", "ZoneResponse",
    "EmailLogItem", "EmailLogsResponse",
    "JobStatus", "JobStatusResponse", "UploadResponse",
    "KPIInfo", "KPISettingsItem", "KPISettingsResponse", "RegisteredKPIsResponse",
    "OrganizationResponse", "OrganizationUpdate",
    "RoleCreate", "RoleListResponse", "RoleResponse", "RoleUpdate",
    "EmailSettingsResponse", "EmailSettingsUpdate",
    "TimezoneSettingsResponse", "TimezoneSettingsUpdate",
    "UserCreate", "UserListResponse", "UserResponse", "UserUpdate",
]
