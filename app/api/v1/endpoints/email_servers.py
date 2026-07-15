"""Email server (SMTP) configuration endpoints — Phase 1.

Implements:
  GET    /api/config/email-servers
  POST   /api/config/email-servers
  GET    /api/config/email-servers/{id}
  PUT    /api/config/email-servers/{id}
  DELETE /api/config/email-servers/{id}   (only when no Role references it)
  POST   /api/config/email-servers/{id}/test   Send a test email

Password is encrypted using app.email_crypto before storage and never returned.

Models used: EmailServer (app.db.models.domain_config)
Schemas: EmailServerCreate, EmailServerUpdate, EmailServerResponse (app.schemas.config)
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/config/email-servers", tags=["config-email-servers"])

# ── Phase 1 — implement below ─────────────────────────────────────────────────
