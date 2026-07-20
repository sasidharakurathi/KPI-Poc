"""Legacy config-file/Configuration-table settings.

GET/PUT /api/settings/email configures SMTP for KPI detection alert emails
only (app.notifications) — a separate system from account/transactional
email (activation, password reset, user onboarding), which is configured via
the EmailServer table (Configuration > Email Servers,
app.api.v1.endpoints.email_servers) instead. See app/notifications.py's
module docstring.
"""
import re

from fastapi import APIRouter, HTTPException

from app import db
from app import notifications
from app.email_crypto import EmailCryptoNotConfigured, encrypt_secret
from app.config_loader import get_all as get_full_config, reload as reload_config
from app.schemas import (
    EmailSettingsResponse, EmailSettingsUpdate,
    TimezoneSettingsResponse, TimezoneSettingsUpdate,
)

router = APIRouter(prefix="/api", tags=["settings"])

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ALLOWED_TIMEZONES = {"Asia/Riyadh", "Asia/Kolkata"}


def _email_config_to_response() -> EmailSettingsResponse:
    cfg = notifications.get_email_config()
    return EmailSettingsResponse(
        enabled=cfg["enabled"],
        smtp_host=cfg["smtp_host"],
        smtp_port=cfg["smtp_port"],
        smtp_username=cfg["smtp_username"],
        use_tls=cfg["use_tls"],
        from_address=cfg["from_address"],
        from_name=cfg["from_name"],
        recipients=cfg["recipients"],
        password_set=bool(cfg["smtp_password_encrypted"]),
        updated_at=db.get_config_updated_at("email"),
    )


@router.get("/config")
async def view_config():
    return get_full_config()


@router.post("/config/reload")
async def hot_reload_config():
    new_cfg = reload_config()
    return {"message": "config.json reloaded", "config": new_cfg}


@router.get("/settings/email", response_model=EmailSettingsResponse)
async def get_email_settings():
    return _email_config_to_response()


@router.put("/settings/email", response_model=EmailSettingsResponse)
async def update_email_settings(body: EmailSettingsUpdate):
    if body.recipients is not None:
        bad = [r for r in body.recipients if not _EMAIL_RE.match(r)]
        if bad:
            raise HTTPException(status_code=422, detail=f"Invalid recipient address(es): {bad}")
    if body.from_address and not _EMAIL_RE.match(body.from_address):
        raise HTTPException(status_code=422, detail="Invalid from_address.")
    if body.smtp_port is not None and not (1 <= body.smtp_port <= 65535):
        raise HTTPException(status_code=422, detail="smtp_port must be between 1 and 65535.")

    updates = body.model_dump(exclude_unset=True, exclude={"smtp_password"})
    if body.smtp_password:
        try:
            updates["smtp_password_encrypted"] = encrypt_secret(body.smtp_password)
        except EmailCryptoNotConfigured as e:
            raise HTTPException(status_code=500, detail=str(e))

    merged = {**notifications.get_email_config(), **updates}
    db.set_config("email", merged)
    return _email_config_to_response()


@router.post("/settings/email/test")
async def test_email_settings():
    cfg = notifications.get_email_config()
    if not cfg["smtp_host"] or not cfg["recipients"]:
        raise HTTPException(
            status_code=400,
            detail="Email settings are incomplete — set SMTP host and at least one recipient first.",
        )
    try:
        notifications.send_test_email(cfg)
    except EmailCryptoNotConfigured as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Test email failed: {e}")
    return {"message": f"Test email sent to {', '.join(cfg['recipients'])}."}


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
