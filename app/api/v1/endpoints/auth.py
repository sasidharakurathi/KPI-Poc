"""Authentication endpoints — Phase 0.

Implements:
  POST /api/auth/register   Organization sign-up (single-tenant first run)
  POST /api/auth/login      Username + password → access + refresh tokens
  POST /api/auth/refresh    Rotate refresh token → new access token
  POST /api/auth/logout     Revoke the current refresh token
  GET  /api/auth/me         Current user profile
  POST /api/auth/forgot-password   Send reset link
  POST /api/auth/reset-password    Consume reset link, set new password

Models used: User, RefreshToken, Organization (app.db.models)
Security helpers: app.core.security (hash_password, verify_password,
                  generate_secure_token, hash_token,
                  create_access_token, decode_access_token)
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/auth", tags=["auth"])

# ── Phase 0 — implement below ─────────────────────────────────────────────────
