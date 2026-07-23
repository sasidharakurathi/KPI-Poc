"""Organization & Site schemas — Phase 0."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


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
    # logo_url: Optional[str] = None
    # Populated on every response whenever the org has a logo (re-read from
    # disk and base64-encoded each time — see
    # app.api.v1.endpoints.organization._to_organization_response), not just
    # the upload response. logo_url is still returned alongside it for
    # direct <img src> use without needing to decode this field.
    logo_base64: Optional[str] = None
    created_at: datetime


class OrganizationUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=120)
    tagline: Optional[str] = Field(default=None, max_length=150)
    default_timezone_id: Optional[str] = None
    site_name: Optional[str] = Field(default=None, min_length=2, max_length=100)
    site_address: Optional[str] = None
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)


class OrganizationLogoUpload(BaseModel):
    """logo_base64 accepts either a plain base64 string or a data URI
    (\"data:image/png;base64,....\") — the data URI prefix, if present, is
    stripped before decoding; the actual image type is then determined by
    sniffing the decoded bytes' own magic number, not trusted from any
    client-declared mime type."""
    logo_base64: str = Field(min_length=1)

    @field_validator("logo_base64")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("logo_base64 must not be blank.")
        return v
