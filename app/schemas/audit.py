"""Audit log schemas — Phase 8.

Field names/values match the frontend's real contract exactly
(`vision-ai-frontend/src/types/domain.ts` AuditLogEntry/AuditEntity/
AuditAction) — entity is "camera" | "kpi_model" | "user" | "role" (no
"config": Priorities/Zones/EmailServers/KpiModelCatalog are explicitly out
of audit-log scope, matching the frontend mock's own collections split).
Configuration (Phase 1) writes are deliberately not audited for the same
reason. "kpi_model" has no current producer — it's Phase 3's KPI Management
capability list, which doesn't exist in this backend yet.

id is a string at the API boundary (DB uses an int PK), matching the
project's established id convention.
"""
from pydantic import BaseModel


class AuditLogEntryResponse(BaseModel):
    id: str
    entity: str           # "camera" | "kpi_model" | "user" | "role"
    entity_id: str
    entity_label: str
    action: str            # "create" | "update" | "delete" | "enable" | "disable"
    actor_username: str
    summary: str
    created_at: str


class AuditLogListResponse(BaseModel):
    count: int
    total: int
    entries: list[AuditLogEntryResponse]
