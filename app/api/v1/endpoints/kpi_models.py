"""KPI model catalog endpoints — Phase 1.

Implements:
  GET    /api/config/kpi-models
  POST   /api/config/kpi-models
  GET    /api/config/kpi-models/{id}
  DELETE /api/config/kpi-models/{id}   (only when no KPI references it)
  PATCH  /api/config/kpi-models/{id}/toggle   Enable / disable

No Edit action by design (PRD §3.5): disable the wrong one, create a new entry.

Models used: KpiModelCatalog (app.db.models.domain_config)
Schemas: KpiModelCreate, KpiModelResponse (app.schemas.config)
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/config/kpi-models", tags=["config-kpi-models"])

# ── Phase 1 — implement below ─────────────────────────────────────────────────
