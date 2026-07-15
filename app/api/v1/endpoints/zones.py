"""Zone configuration endpoints — Phase 1.

Implements:
  GET    /api/config/zones
  POST   /api/config/zones
  PUT    /api/config/zones/{id}
  DELETE /api/config/zones/{id}   (only when no cameras reference it)
  PATCH  /api/config/zones/{id}/toggle

Models used: Zone (app.db.models.domain_config)
Schemas: ZoneCreate, ZoneResponse (app.schemas.config)
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/config/zones", tags=["config-zones"])

# ── Phase 1 — implement below ─────────────────────────────────────────────────
