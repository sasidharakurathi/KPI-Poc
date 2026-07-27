"""Timezone catalog schema - Phase 1. Read-only: no Create/Update/Delete
schemas exist because this catalog is static, seeded once at startup
(app/db/seed_data/timezones.py) and never modified via the API."""
from typing import Optional

from pydantic import BaseModel


class TimezoneResponse(BaseModel):
    model_config = {"from_attributes": True}
    id: int
    abbreviation: str
    timezone_name: str
    utc_offset: Optional[str] = None
    gmt_offset: Optional[str] = None
