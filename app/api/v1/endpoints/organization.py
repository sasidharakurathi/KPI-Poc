"""Organization settings endpoints — Phase 0.

Implements:
  GET  /api/organization     Get current organization details
  PUT  /api/organization     Update name, tagline, timezone, site details (Owner/Admin only)

Path is "/api/organization" (singular, matching the frontend's
src/api/organization.ts contract) — an earlier stub draft assumed "/api/org".

Per the confirmed Phase 0 decision, there is no separate Site entity: the
Organization row carries its site fields (site_name/site_address/latitude/
longitude) directly.

Models used: Organization (app.db.models.organization)
Schemas: OrganizationResponse, OrganizationUpdate (app.schemas.organization)
"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import select

from app.core.dependencies import DbSession, require_permission
from app.db.models import Organization
from app.schemas.organization import OrganizationResponse, OrganizationUpdate

router = APIRouter(prefix="/api/organization", tags=["organization"])


def _get_org(db: DbSession) -> Organization:
    org = db.exec(select(Organization)).first()
    if org is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No organization has been set up yet.")
    return org


@router.get("", response_model=OrganizationResponse)
def get_organization(
    db: DbSession,
    _user: dict = Depends(require_permission("organization_settings", "view")),
) -> Organization:
    return _get_org(db)


@router.put("", response_model=OrganizationResponse)
def update_organization(
    payload: OrganizationUpdate,
    db: DbSession,
    _user: dict = Depends(require_permission("organization_settings", "edit")),
) -> Organization:
    org = _get_org(db)
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(org, key, value)
    org.updated_at = datetime.utcnow()
    db.add(org)
    db.commit()
    db.refresh(org)
    return org
