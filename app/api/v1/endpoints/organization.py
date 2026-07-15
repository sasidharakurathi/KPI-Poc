"""Organization settings endpoints — Phase 0.

Implements:
  GET  /api/org     Get current organization details
  PUT  /api/org     Update name, tagline, timezone, site details (Owner/Admin only)

Models used: Organization (app.db.models.organization)
Schemas: OrganizationResponse, OrganizationUpdate (app.schemas.organization)
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/org", tags=["organization"])

# ── Phase 0 — implement below ─────────────────────────────────────────────────
