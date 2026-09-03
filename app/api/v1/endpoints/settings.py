"""Legacy config-file/Configuration-table settings.

SMTP for KPI detection alert emails is no longer configured here - it's
resolved live from each organization's active EmailServer row (Configuration
> Email Servers, app.api.v1.endpoints.email_servers) - see
app/notifications.py's module docstring.
"""
from fastapi import APIRouter, HTTPException

from app import db
from app.config_loader import get_all as get_full_config, reload as reload_config
from app.schemas import TimezoneSettingsResponse, TimezoneSettingsUpdate

router = APIRouter(prefix="/api", tags=["settings"])

_ALLOWED_TIMEZONES = {"Asia/Riyadh", "Asia/Kolkata"}


@router.get("/config")
async def view_config():
    return get_full_config()


@router.post("/config/reload")
async def hot_reload_config():
    new_cfg = reload_config()
    return {"message": "config.json reloaded", "config": new_cfg}


@router.get("/settings/timezone", response_model=TimezoneSettingsResponse)
async def get_timezone_settings():
    cfg = db.get_config("timezone") or {}
    return TimezoneSettingsResponse(
        default=cfg.get("default", "Asia/Riyadh"),
        updated_at=db.get_config_updated_at("timezone"),
    )


@router.put("/settings/timezone", response_model=TimezoneSettingsResponse)
async def update_timezone_settings(body: TimezoneSettingsUpdate):
    if body.default not in _ALLOWED_TIMEZONES:
        raise HTTPException(
            status_code=422,
            detail=f"default must be one of {sorted(_ALLOWED_TIMEZONES)}.",
        )
    db.set_config("timezone", {"default": body.default})
    return TimezoneSettingsResponse(
        default=body.default,
        updated_at=db.get_config_updated_at("timezone"),
    )
