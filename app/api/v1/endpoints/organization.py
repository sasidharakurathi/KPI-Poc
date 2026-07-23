"""Organization settings endpoints — Phase 0.

Implements:
  GET  /api/organization         Get the caller's own organization
  PUT  /api/organization         Update name, tagline, timezone, site details (Owner/Admin only)
  POST /api/organization/logo    Create/replace the organization's logo, via base64 JSON (Owner/Admin only)
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

The logo endpoint takes base64 (JSON body), not a multipart file upload —
simpler for the frontend (no separate FormData request). The decoded image
is saved to disk under settings.LOGOS_DIR for persistence, and read back as
base64 on every response (logo_base64) — there is no logo_url; callers get
the image data directly rather than a link to fetch it separately. The image
type is determined by sniffing the decoded bytes' own magic number rather
than trusting a client-declared content type.

Models used: Organization (app.db.models.organization)
Schemas: OrganizationResponse, OrganizationUpdate, OrganizationLogoUpload (app.schemas.organization)
"""
import base64
import binascii
import logging
import secrets
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import select

from app.config import settings
from app.core.dependencies import DbSession, require_auth, require_permission
from app.db.models import Organization
from app.schemas.organization import OrganizationLogoUpload, OrganizationResponse, OrganizationUpdate
from app.services.auth_service import get_timezone_or_422

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/organization", tags=["organization"])
organizations_router = APIRouter(prefix="/api/organizations", tags=["organization"])

# Signature-sniffing, not a client-declared content type — the first rule
# whose bytes match wins. SVG is checked separately (it's text, not a fixed
# binary magic number) after none of these match.
_MAGIC_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
]


def _sniff_image_extension(data: bytes) -> str | None:
    for signature, ext in _MAGIC_SIGNATURES:
        if data.startswith(signature):
            return ext
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    head = data[:512].decode("utf-8", errors="ignore").lstrip().lower()
    if head.startswith("<?xml") or head.startswith("<svg"):
        if "<svg" in head:
            return ".svg"
    return None


def _decode_base64_image(raw: str) -> bytes:
    payload = raw.split(",", 1)[1] if raw.strip().startswith("data:") else raw
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "logo_base64 is not valid base64.")


def _get_own_org(db: DbSession, user: dict) -> Organization:
    org = db.get(Organization, user.get("org_id"))
    if org is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No organization has been set up yet.")
    return org


def _to_organization_response(org: Organization) -> OrganizationResponse:
    logo_base64 = None
    if org.logo_path:
        try:
            logo_base64 = base64.b64encode((settings.LOGOS_DIR / org.logo_path).read_bytes()).decode("ascii")
        except OSError:
            logger.warning("[organization] could not read back logo file %s for base64 response", org.logo_path)
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
        logo_base64=logo_base64,
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
def upload_organization_logo(
    payload: OrganizationLogoUpload,
    db: DbSession,
    user: dict = Depends(require_permission("organization_settings", "edit")),
) -> OrganizationResponse:
    """Creates the org's logo if it doesn't have one yet, or replaces the
    existing one — same endpoint either way, matching how PUT /api/organization
    handles create-vs-update for every other field. The old file (if any) is
    deleted from disk after the new one is saved."""
    org = _get_own_org(db, user)

    content = _decode_base64_image(payload.logo_base64)
    if len(content) > settings.LOGO_MAX_SIZE_BYTES:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Logo image too large — max {settings.LOGO_MAX_SIZE_BYTES // (1024 * 1024)} MB.",
        )
    ext = _sniff_image_extension(content)
    if ext is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Unrecognized image data. Supported formats: PNG, JPEG, WEBP, SVG.",
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
