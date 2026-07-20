"""Assembles all v1 routers into a single APIRouter included by app.main."""
from fastapi import APIRouter

from .endpoints import (
    alerts,
    audit,
    auth,
    cameras,
    dashboard,
    email_logs,
    email_servers,
    health,
    kpi_models,
    kpis,
    organization,
    priorities,
    roles,
    settings,
    timezones,
    users,
    videos,
    zones,
)

api_router = APIRouter()

# ── Active (existing functionality) ──────────────────────────────────────────
api_router.include_router(health.router)
api_router.include_router(kpis.router)
api_router.include_router(cameras.router)
api_router.include_router(alerts.router)
api_router.include_router(videos.router)
api_router.include_router(settings.router)
api_router.include_router(email_logs.router)

# ── Phase stubs (routers registered but empty — teams add endpoints here) ─────
api_router.include_router(auth.router)           # Phase 0
api_router.include_router(organization.router)   # Phase 0
api_router.include_router(priorities.router)     # Phase 1
api_router.include_router(zones.router)          # Phase 1
api_router.include_router(email_servers.router)  # Phase 1
api_router.include_router(kpi_models.router)     # Phase 1
api_router.include_router(timezones.router)      # Phase 1 (static catalog, public)
api_router.include_router(roles.router)          # Phase 6
api_router.include_router(users.router)          # Phase 7
api_router.include_router(dashboard.router)      # Phase 5
api_router.include_router(audit.router)          # Phase 8
