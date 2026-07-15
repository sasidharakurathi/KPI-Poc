"""Role & permissions endpoints — Phase 6.

Implements:
  GET    /api/roles
  POST   /api/roles
  GET    /api/roles/{id}
  PUT    /api/roles/{id}
  DELETE /api/roles/{id}   (only when no active user holds it)

The Owner role (is_system=True) cannot be deleted or have permissions modified.

Models used: Role (app.db.models.role)
Schemas: RoleCreate, RoleUpdate, RoleResponse, RoleListResponse (app.schemas.role)
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/roles", tags=["roles"])

# ── Phase 6 — implement below ─────────────────────────────────────────────────
