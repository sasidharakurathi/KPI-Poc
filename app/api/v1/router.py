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
    kpi_labels,
    kpis,
    organization,
    priorities,
    roles,
    settings,
    timezones,
    users,
    videos,
    ws,
    zones,
)

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(kpis.router)
api_router.include_router(cameras.router)
api_router.include_router(kpi_labels.router)
api_router.include_router(alerts.router)
api_router.include_router(videos.router)
api_router.include_router(settings.router)
api_router.include_router(email_logs.router)
api_router.include_router(auth.router)                        
api_router.include_router(organization.router)                
api_router.include_router(organization.organizations_router)  
api_router.include_router(priorities.router)     
api_router.include_router(zones.router)          
api_router.include_router(email_servers.router)
api_router.include_router(timezones.router)
api_router.include_router(roles.router)          
api_router.include_router(users.router)          
api_router.include_router(dashboard.router)      
api_router.include_router(audit.router)          
api_router.include_router(ws.router)             
