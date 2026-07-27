"""Business logic for Phase 8: Audit Log.

log_action() is called from every Camera/Role/User mutation (create/update/
delete/enable/disable) in app.services.camera_service/role_service/
user_service. It is a best-effort side effect exactly like the websocket
broadcast in app.db.create_alert or the stream sync in camera_service: a
failure to write an audit entry must never break the actual mutation it
describes, so it swallows and logs its own errors rather than raising.

No zone-scoping on the read side (list_audit_logs/GET /api/audit) - Role and
User entries have no zone concept to scope by at all, so a partial
(Camera-only) restriction would be inconsistent. Visibility is gated purely
by require_permission("audit_log", "view"), matching how audit access is
normally an admin-only concern rather than a per-zone one.
"""
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import func as _func
from sqlmodel import Session, select

from app.db.models import AuditLog
from app.schemas.audit import AuditLogEntryResponse, AuditLogListResponse

logger = logging.getLogger(__name__)


def log_action(
    db: Session, *,
    entity: str, entity_id: Optional[str], entity_label: Optional[str],
    action: str, summary: str,
    actor_id: Optional[int], actor_name: Optional[str],
    before: Optional[dict] = None, after: Optional[dict] = None,
) -> None:
    try:
        entry = AuditLog(
            entity_type=entity, entity_id=entity_id, entity_name=entity_label,
            action=action, summary=summary,
            before=before, after=after,
            actor_id=actor_id, actor_name=actor_name,
        )
        db.add(entry)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("[audit] failed to log %s.%s for entity_id=%s", entity, action, entity_id)


def _to_entry_response(row: AuditLog) -> AuditLogEntryResponse:
    return AuditLogEntryResponse(
        id=str(row.id),
        entity=row.entity_type,
        entity_id=row.entity_id or "",
        entity_label=row.entity_name or row.entity_id or "",
        action=row.action,
        actor_username=row.actor_name or "system",
        summary=row.summary or "",
        created_at=row.created_at.isoformat(),
    )


_SORTABLE = {"created_at": AuditLog.created_at}


def list_audit_logs(
    db: Session, *,
    entity: Optional[str] = None, action: Optional[str] = None, actor: Optional[str] = None,
    date_from: Optional[datetime] = None, date_to: Optional[datetime] = None,
    sort_dir: str = "desc", limit: int = 100, offset: int = 0,
) -> AuditLogListResponse:
    stmt = select(AuditLog)
    count_stmt = select(_func.count()).select_from(AuditLog)

    if entity:
        stmt = stmt.where(AuditLog.entity_type == entity)
        count_stmt = count_stmt.where(AuditLog.entity_type == entity)
    if action:
        stmt = stmt.where(AuditLog.action == action)
        count_stmt = count_stmt.where(AuditLog.action == action)
    if actor:
        stmt = stmt.where(AuditLog.actor_name.ilike(f"%{actor}%"))
        count_stmt = count_stmt.where(AuditLog.actor_name.ilike(f"%{actor}%"))
    if date_from:
        stmt = stmt.where(AuditLog.created_at >= date_from)
        count_stmt = count_stmt.where(AuditLog.created_at >= date_from)
    if date_to:
        stmt = stmt.where(AuditLog.created_at <= date_to)
        count_stmt = count_stmt.where(AuditLog.created_at <= date_to)

    total = db.exec(count_stmt).one()

    col = AuditLog.created_at
    stmt = stmt.order_by(col.desc() if sort_dir == "desc" else col.asc())
    stmt = stmt.offset(offset).limit(limit)

    rows = db.exec(stmt).all()
    return AuditLogListResponse(
        count=len(rows), total=total, entries=[_to_entry_response(r) for r in rows],
    )
