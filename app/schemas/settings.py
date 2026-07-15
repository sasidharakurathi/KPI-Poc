from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class EmailSettingsResponse(BaseModel):
    enabled: bool
    smtp_host: str
    smtp_port: int
    smtp_username: str
    use_tls: bool
    from_address: str
    from_name: str
    recipients: list[str]
    password_set: bool
    updated_at: Optional[datetime] = None


class EmailSettingsUpdate(BaseModel):
    enabled: Optional[bool] = None
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    use_tls: Optional[bool] = None
    from_address: Optional[str] = None
    from_name: Optional[str] = None
    recipients: Optional[list[str]] = None


class TimezoneSettingsResponse(BaseModel):
    default: str
    updated_at: Optional[datetime] = None


class TimezoneSettingsUpdate(BaseModel):
    default: str
