"""Organization & Site schemas — Phase 0."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class OrganizationResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    org_id: str
    name: str
    tagline: Optional[str] = None
    default_timezone_id: Optional[str] = None
    site_name: Optional[str] = None
    site_address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    logo_url: Optional[str] = None
    created_at: datetime


class OrganizationUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=120)
    tagline: Optional[str] = Field(default=None, max_length=150)
    default_timezone_id: Optional[str] = None
    site_name: Optional[str] = Field(default=None, min_length=2, max_length=100)
    site_address: Optional[str] = None
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
