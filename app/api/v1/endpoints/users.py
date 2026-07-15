"""User management endpoints — Phase 7.

Implements:
  GET    /api/users
  POST   /api/users           Create user; sends temp-password email
  GET    /api/users/{id}
  PUT    /api/users/{id}      Edit role, address, phone, designation, description
  PATCH  /api/users/{id}/disable
  DELETE /api/users/{id}      Soft-delete (hidden from lists, history retained)

Models used: User (app.db.models.user)
Schemas: UserCreate, UserUpdate, UserResponse, UserListResponse (app.schemas.user)
Security: app.core.security (hash_password, generate_secure_token)
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/users", tags=["users"])

# ── Phase 7 — implement below ─────────────────────────────────────────────────
