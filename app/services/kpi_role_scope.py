"""Shared KPI-visibility scoping - the KPI-alert counterpart to
app.services.zone_scope's zone-visibility scoping.

Used by Alerts, the realtime websocket, and Dashboard to restrict what a
KPI-restricted role (Role.kpi_names non-empty) can see, and by
app.notifications to decide who gets emailed for a given alert. None =
unrestricted (sees/receives everything for this org, including "system"
connectivity alerts). A non-None set restricts to alerts whose kpi_name is in
that set - "system" (camera-offline/connectivity alerts, not a real
detector) is always included in a restricted set, since those aren't
KPI-specific and every role that can see alerts at all should see them. The
built-in Owner role and any role with an empty kpi_names list are always
unrestricted.
"""
from typing import Optional

from sqlmodel import Session, select

from app.db.models import Role, User

_ALWAYS_ALLOWED_KPI_NAMES = {"system"}


def allowed_kpi_names_for_role(role: Optional[Role]) -> Optional[set[str]]:
    """Takes an already-resolved Role (or None) - for callers that need the
    role object for other checks too and don't want to look it up twice."""
    if role and role.is_system:
        return None
    kpi_names = (role.kpi_names if role else None) or []
    if not kpi_names:
        return None
    return set(kpi_names) | _ALWAYS_ALLOWED_KPI_NAMES


def allowed_kpi_names_for_user(user: dict, db: Session) -> Optional[set[str]]:
    """Convenience wrapper for the require_permission dependency's user dict -
    resolves the role itself, then delegates to allowed_kpi_names_for_role."""
    if user.get("is_system"):
        return None
    role_id = user.get("role_id")
    role = db.get(Role, role_id) if role_id else None
    return allowed_kpi_names_for_role(role)


def _has_alerts_view(role: Role) -> bool:
    return "view" in (role.permissions or {}).get("alerts", [])


def eligible_recipients_for_alert(db: Session, org_id: Optional[int], kpi_name: str) -> list[str]:
    """Active users in this org whose role can see this alert: the Owner
    role, or any role with alerts:view permission that isn't restricted away
    from this kpi_name (unrestricted = empty kpi_names, or kpi_name=="system").
    Returns deduplicated login_email addresses."""
    if org_id is None:
        return []

    rows = db.exec(
        select(User, Role)
        .join(Role, User.role_id == Role.id)
        .where(User.org_id == org_id, User.status == "active")
    ).all()

    recipients: list[str] = []
    seen: set[str] = set()
    for user, role in rows:
        if not user.login_email:
            continue
        allowed = role.is_system or (
            _has_alerts_view(role)
            and (kpi_name in _ALWAYS_ALLOWED_KPI_NAMES or not role.kpi_names or kpi_name in role.kpi_names)
        )
        if allowed and user.login_email not in seen:
            seen.add(user.login_email)
            recipients.append(user.login_email)
    return recipients
