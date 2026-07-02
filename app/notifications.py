"""Email alert notifications — global SMTP config lives in the generic
`configurations` table under name="email" (see app.db.get_config/set_config).

notify_alert() is fire-and-forget (runs on a background thread) so a slow or
unreachable mail server never blocks KPI processing. send_test_email() is
synchronous so the admin UI's "Send test" action can surface failures directly.
Every attempt (success or failure) is recorded in the email_logs table.
"""
import logging
import smtplib
import ssl
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Optional

from . import db
from .email_crypto import decrypt_secret

logger = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "smtp_host": "",
    "smtp_port": 587,
    "smtp_username": "",
    "smtp_password_encrypted": None,
    "use_tls": True,
    "from_address": "",
    "from_name": "Vision AI Alerts",
    "recipients": [],
}


def get_email_config() -> dict[str, Any]:
    """Stored config merged over defaults — always returns a complete dict."""
    stored = db.get_config("email") or {}
    return {**_DEFAULTS, **stored}


def _build_message(cfg: dict, subject: str, text: str, html: str) -> MIMEMultipart:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{cfg['from_name']} <{cfg['from_address']}>" if cfg["from_name"] else cfg["from_address"]
    msg["To"] = ", ".join(cfg["recipients"])
    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))
    return msg


def _send(cfg: dict, msg: MIMEMultipart) -> None:
    password = decrypt_secret(cfg["smtp_password_encrypted"]) if cfg["smtp_password_encrypted"] else ""
    context = ssl.create_default_context()

    if cfg["use_tls"]:
        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"], timeout=10) as server:
            server.starttls(context=context)
            if cfg["smtp_username"]:
                server.login(cfg["smtp_username"], password)
            server.sendmail(cfg["from_address"], cfg["recipients"], msg.as_string())
    else:
        with smtplib.SMTP_SSL(cfg["smtp_host"], cfg["smtp_port"], timeout=10, context=context) as server:
            if cfg["smtp_username"]:
                server.login(cfg["smtp_username"], password)
            server.sendmail(cfg["from_address"], cfg["recipients"], msg.as_string())


def notify_alert(
    kpi_name: str,
    display_name: str,
    alert_type: str,
    job_id: str,
    alert_id: int,
    confidence: float,
) -> None:
    """Send an email for a newly-saved alert, if email is enabled/configured.
    Never raises — failures are logged, not propagated to the KPI pipeline."""
    cfg = get_email_config()
    if not cfg["enabled"] or not cfg["recipients"] or not cfg["smtp_host"]:
        return

    job = db.get_job(job_id)
    camera_id = job.camera_id if job else None
    camera_name = job.camera_name if job else None

    subject = f"[Vision AI] {display_name} alert" + (f" — {camera_name}" if camera_name else "")
    text = (
        f"KPI: {display_name}\n"
        f"Alert type: {alert_type}\n"
        f"Camera: {camera_name or 'N/A'}\n"
        f"Job ID: {job_id}\n"
        f"Alert ID: {alert_id}\n"
    )
    html = (
        f"<h3>{display_name} alert</h3>"
        f"<p><b>Alert type:</b> {alert_type}<br>"
        f"<b>Camera:</b> {camera_name or 'N/A'}<br>"
        f"<b>Job ID:</b> {job_id}<br>"
        f"<b>Alert ID:</b> {alert_id}</p>"
    )
    msg = _build_message(cfg, subject, text, html)

    def _worker() -> None:
        try:
            _send(cfg, msg)
            logger.info(f"[email] sent notification for alert #{alert_id} ({kpi_name})")
            db.create_email_log(
                status="sent", subject=subject, recipients=cfg["recipients"],
                alert_id=alert_id, kpi_name=kpi_name, alert_type=alert_type,
                camera_id=camera_id, camera_name=camera_name,
            )
        except Exception as exc:
            logger.exception(f"[email] failed to send notification for alert #{alert_id} ({kpi_name})")
            db.create_email_log(
                status="failed", subject=subject, recipients=cfg["recipients"],
                alert_id=alert_id, kpi_name=kpi_name, alert_type=alert_type,
                camera_id=camera_id, camera_name=camera_name, error=str(exc),
            )

    threading.Thread(target=_worker, daemon=True, name="email-notify").start()


def send_test_email(cfg: Optional[dict] = None) -> None:
    """Synchronous send used by the admin UI's test button — raises on failure.
    Logged the same as a real alert notification, tagged kpi_name=None."""
    cfg = cfg or get_email_config()
    text = (
        "This is a test email from your Vision AI Alerts configuration.\n"
        "If you received this, SMTP is configured correctly."
    )
    html = f"<p>{text}</p>"
    subject = "[Vision AI] Test email"
    msg = _build_message(cfg, subject, text, html)
    try:
        _send(cfg, msg)
        db.create_email_log(status="sent", subject=subject, recipients=cfg["recipients"], alert_type="test")
    except Exception as exc:
        db.create_email_log(status="failed", subject=subject, recipients=cfg["recipients"], alert_type="test", error=str(exc))
        raise
