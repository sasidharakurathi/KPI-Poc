from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TimezoneSettingsResponse(BaseModel):
    default: str
    updated_at: Optional[datetime] = None


class TimezoneSettingsUpdate(BaseModel):
    default: str
