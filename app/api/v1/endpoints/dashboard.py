"""Dashboard aggregation endpoints — Phase 5.

Implements:
  GET /api/dashboard/summary        Alert counts, camera status summary
  GET /api/dashboard/alert-chart    Alert counts per KPI over a date range for a camera
  GET /api/dashboard/cameras        Camera list with live connectivity status

These endpoints aggregate data from the existing alerts/cameras tables — no new
tables needed. They are read-only and safe to implement independently of other phases.
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# ── Phase 5 — implement below ─────────────────────────────────────────────────
