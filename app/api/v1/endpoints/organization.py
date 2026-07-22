"""Organization settings endpoints — Phase 0.

Implements:
  GET  /api/organization         Get the caller's own organization
  PUT  /api/organization         Update name, tagline, timezone, site details (Owner/Admin only)
  POST /api/organization/logo    Upload/replace the organization's logo (Owner/Admin only)
  GET  /api/organizations        List every organization on this deployment (Owner-role only)

This is a multi-tenant deployment: any number of organizations can exist.
GET/PUT "/api/organization" (singular) always operate on the caller's own
org, resolved from their JWT — never from a client-supplied id, exactly like
every other org-scoped resource in this codebase. "/api/organizations"
(plural) is the one deliberate exception — it crosses tenant boundaries by
design, so it's gated more strictly (is_system/Owner role, not just
organization_settings permission, which is granted per-org and would
otherwise let one org's admin enumerate every other org).

Per the confirmed Phase 0 decision, there is no separate Site entity: the
Organization row carries its site fields (site_name/site_address/latitude/
longitude) directly.

Models used: Organization (app.db.models.organization)
Schemas: OrganizationResponse, OrganizationUpdate (app.schemas.organization)
"""
import logging
import secrets
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlmodel import select

from app.config import settings
from app.core.dependencies import DbSession, require_auth, require_permission
from app.db.models import Organization
from app.schemas.organization import OrganizationResponse, OrganizationUpdate
from app.services.auth_service import get_timezone_or_422

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/organization", tags=["organization"])
organizations_router = APIRouter(prefix="/api/organizations", tags=["organization"])

_ALLOWED_LOGO_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
}


def _get_own_org(db: DbSession, user: dict) -> Organization:
    org = db.get(Organization, user.get("org_id"))
    if org is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No organization has been set up yet.")
    return org


def _to_organization_response(org: Organization) -> OrganizationResponse:
    logo_url = f"{settings.PUBLIC_BASE_URL}/media/logos/{org.logo_path}" if org.logo_path else None
    return OrganizationResponse(
        id=org.id,
        org_id=org.org_id,
        name=org.name,
        tagline=org.tagline,
        default_timezone_id=str(org.default_timezone_id) if org.default_timezone_id is not None else None,
        site_name=org.site_name,
        site_address=org.site_address,
        latitude=org.latitude,
        longitude=org.longitude,
        logo_url=logo_url,
        created_at=org.created_at,
    )


@router.get("", response_model=OrganizationResponse)
def get_organization(
    db: DbSession,
    user: dict = Depends(require_permission("organization_settings", "view")),
) -> OrganizationResponse:
    return _to_organization_response(_get_own_org(db, user))


@router.put("", response_model=OrganizationResponse)
def update_organization(
    payload: OrganizationUpdate,
    db: DbSession,
    user: dict = Depends(require_permission("organization_settings", "edit")),
) -> OrganizationResponse:
    org = _get_own_org(db, user)
    updates = payload.model_dump(exclude_unset=True)
    if "default_timezone_id" in updates:
        raw = updates.pop("default_timezone_id")
        updates["default_timezone_id"] = get_timezone_or_422(db, raw).id if raw is not None else None
    for key, value in updates.items():
        setattr(org, key, value)
    org.updated_at = datetime.utcnow()
    db.add(org)
    db.commit()
    db.refresh(org)
    return _to_organization_response(org)


@router.post("/logo", response_model=OrganizationResponse)
async def upload_organization_logo(
    db: DbSession,
    file: UploadFile,
    user: dict = Depends(require_permission("organization_settings", "edit")),
) -> OrganizationResponse:
    org = _get_own_org(db, user)

    ext = _ALLOWED_LOGO_TYPES.get(file.content_type)
    if ext is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Unsupported logo type '{file.content_type}'. Allowed: "
            f"{', '.join(sorted(_ALLOWED_LOGO_TYPES))}.",
        )

    content = await file.read()
    if len(content) > settings.LOGO_MAX_SIZE_BYTES:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Logo file too large — max {settings.LOGO_MAX_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    old_filename = org.logo_path
    filename = f"org_{org.id}_{secrets.token_hex(4)}{ext}"
    settings.LOGOS_DIR.mkdir(parents=True, exist_ok=True)
    (settings.LOGOS_DIR / filename).write_bytes(content)

    org.logo_path = filename
    org.updated_at = datetime.utcnow()
    db.add(org)
    db.commit()
    db.refresh(org)

    if old_filename:
        old_path = settings.LOGOS_DIR / old_filename
        try:
            old_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("[organization] failed to remove old logo file %s", old_path)

    return _to_organization_response(org)


@organizations_router.get("", response_model=list[OrganizationResponse], summary="List Organizations")
def list_organizations(
    db: DbSession,
    user: dict = Depends(require_auth),
) -> list[OrganizationResponse]:
    """Every organization on this deployment — crosses tenant boundaries by
    design, so it's restricted to callers whose role is_system=True (the
    built-in Owner role), not merely organization_settings.view (which is
    granted per-org and would otherwise let one org's admin enumerate every
    other org on the platform)."""
    if not user.get("is_system"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only an organization Owner can list every organization.")
    orgs = db.exec(select(Organization).order_by(Organization.id)).all()
    return [_to_organization_response(org) for org in orgs]
