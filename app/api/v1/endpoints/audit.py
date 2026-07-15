"""Audit log endpoints — Phase 8.

Implements:
  GET /api/audit   Read-only log with filters: entity_type, date range, actor

Writing to audit_logs is done via app.services.audit_service.log_action(), which
should be called from every endpoint that creates, updates, deletes, enables, or
disables a Camera, KPI, User, or Role.

Models used: AuditLog (app.db.models.audit)
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/audit", tags=["audit"])

# ── Phase 8 — implement below ─────────────────────────────────────────────────
