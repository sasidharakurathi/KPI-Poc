"""Priority configuration endpoints — Phase 1.

Implements:
  GET    /api/config/priorities           List all priorities
  POST   /api/config/priorities           Create a priority
  PUT    /api/config/priorities/{id}      Update name/color
  DELETE /api/config/priorities/{id}      Delete (only if no cameras or KPIs reference it)
  PATCH  /api/config/priorities/{id}/toggle   Enable / disable

Models used: Priority (app.db.models.domain_config)
Schemas: PriorityCreate, PriorityResponse (app.schemas.config)
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/config/priorities", tags=["config-priorities"])

# ── Phase 1 — implement below ─────────────────────────────────────────────────
