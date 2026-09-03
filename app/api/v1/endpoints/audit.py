"""Audit log endpoints - Phase 8.

Implements:
  GET /api/audit   Read-only log with filters: entity, action, actor, date range

Writing to audit_logs is done via app.services.audit_service.log_action(),
called from every Camera/Role/User create/update/delete/enable/disable
(app.services.camera_service/role_service/user_service). Priorities/Zones/
EmailServers (Phase 1) are deliberately not audited, matching the frontend's
own audit-log scope.

No zone-scoping - require_permission("audit_log", "view") is the only gate
(see app.services.audit_service's module docstring for why).

Models used: AuditLog (app.db.models.audit)
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import DbSession, require_permission
from app.schemas.audit import AuditLogListResponse
from app.services import audit_service

router = APIRouter(prefix="/api/audit", tags=["audit"])


def _parse_date(d: Optional[str]) -> Optional[datetime]:
    if not d:
        return None
    try:
        return datetime.fromisoformat(d)
    except ValueError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Invalid date '{d}' - expected ISO 8601.")


@router.get("", response_model=AuditLogListResponse)
async def list_audit_log(
    db: DbSession,
    user: dict = Depends(require_permission("audit_log", "view")),
    entity: Optional[str] = None,
    action: Optional[str] = None,
    actor: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort_dir: str = "desc",
    limit: int = 100,
    offset: int = 0,
):
    return audit_service.list_audit_logs(
        db, entity=entity, action=action, actor=actor,
        date_from=_parse_date(date_from), date_to=_parse_date(date_to),
        sort_dir=sort_dir, limit=limit, offset=offset,
    )
