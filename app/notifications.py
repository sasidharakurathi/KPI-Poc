
import logging
import smtplib
import ssl
import threading
from datetime import datetime, timezone
from email.mime.image import MIMEImage
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

# Maps internal alert_type slugs to friendly display labels.
_ALERT_LABELS: dict[str, str] = {
    "phone_usage_confirmed": "Mobile Phone Usage Detected",
    "smoking_alarm":         "Smoking Detected",
    "ppe_non_compliance":    "PPE Non-Compliance Detected",
    "fire_detected":         "Fire Detected",
    "smoke_detected":        "Smoke Detected",
    "fire_smoke_detected":   "Fire & Smoke Detected",
    "fall_detected":         "Person Fall Detected",
    "person_detected":       "Person Detected",
    "carton_box_detected":   "Carton Box Detected",
    "anpr_plate_detected":   "Number Plate Detected",
}

# Accent colour per alert type (used in the email header band).
_ALERT_COLORS: dict[str, str] = {
    "phone_usage_confirmed": "#1565C0",
    "smoking_alarm":         "#6A1B9A",
    "ppe_non_compliance":    "#E65100",
    "fire_detected":         "#B71C1C",
    "smoke_detected":        "#37474F",
    "fire_smoke_detected":   "#BF360C",
    "fall_detected":         "#AD1457",
    "person_detected":       "#1B5E20",
    "carton_box_detected":   "#F57F17",
    "anpr_plate_detected":   "#0D47A1",
}
_DEFAULT_COLOR = "#1A237E"


def get_email_config() -> dict[str, Any]:
    """Stored config merged over defaults — always returns a complete dict."""
    stored = db.get_config("email") or {}
    return {**_DEFAULTS, **stored}


def _humanize(alert_type: str) -> str:
    return _ALERT_LABELS.get(alert_type, alert_type.replace("_", " ").title())


def _build_html(
    display_name: str,
    alert_type: str,
    camera_name: Optional[str],
    job_id: str,
    alert_id: int,
    timestamp: str,
    has_image: bool,
) -> str:
    label  = _humanize(alert_type)
    color  = _ALERT_COLORS.get(alert_type, _DEFAULT_COLOR)
    cam    = camera_name or "Unknown Camera"
    img_block = (
        '<tr><td style="padding:0 32px 24px;">'
        '<img src="cid:alert_frame" alt="Detection frame" '
        'style="width:100%;max-width:580px;border-radius:8px;display:block;'
        'border:1px solid #E0E0E0;" /></td></tr>'
    ) if has_image else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#F5F5F5;font-family:Arial,Helvetica,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#F5F5F5;padding:32px 0;">
  <tr><td align="center">
    <table width="620" cellpadding="0" cellspacing="0"
           style="background:#FFFFFF;border-radius:12px;overflow:hidden;
                  box-shadow:0 2px 8px rgba(0,0,0,0.08);">

      <!-- Header band -->
      <tr>
        <td style="background:{color};padding:28px 32px;">
          <p style="margin:0;color:rgba(255,255,255,0.75);font-size:12px;
                    letter-spacing:1px;text-transform:uppercase;">JANA Vision AI Monitoring</p>
          <h1 style="margin:6px 0 0;color:#FFFFFF;font-size:22px;font-weight:700;
                     line-height:1.3;">{label}</h1>
        </td>
      </tr>

      <!-- Meta info -->
      <tr>
        <td style="padding:24px 32px 20px;">
          <table width="100%" cellpadding="0" cellspacing="0">
            <tr>
              <td style="width:50%;padding-bottom:12px;vertical-align:top;">
                <p style="margin:0;font-size:11px;color:#9E9E9E;text-transform:uppercase;
                          letter-spacing:0.8px;">Camera</p>
                <p style="margin:4px 0 0;font-size:15px;color:#212121;font-weight:600;">{cam}</p>
              </td>
              <td style="width:50%;padding-bottom:12px;vertical-align:top;">
                <p style="margin:0;font-size:11px;color:#9E9E9E;text-transform:uppercase;
                          letter-spacing:0.8px;">Detection Type</p>
                <p style="margin:4px 0 0;font-size:15px;color:#212121;font-weight:600;">{display_name}</p>
              </td>
            </tr>
            <tr>
              <td style="padding-bottom:4px;vertical-align:top;">
                <p style="margin:0;font-size:11px;color:#9E9E9E;text-transform:uppercase;
                          letter-spacing:0.8px;">Time</p>
                <p style="margin:4px 0 0;font-size:15px;color:#212121;font-weight:600;">{timestamp}</p>
              </td>
              <td style="padding-bottom:4px;vertical-align:top;">
                <p style="margin:0;font-size:11px;color:#9E9E9E;text-transform:uppercase;
                          letter-spacing:0.8px;">Alert ID</p>
                <p style="margin:4px 0 0;font-size:15px;color:#212121;font-weight:600;">#{alert_id}</p>
              </td>
            </tr>
          </table>
        </td>
      </tr>

      <!-- Divider -->
      <tr><td style="padding:0 32px;">
        <hr style="border:none;border-top:1px solid #EEEEEE;margin:0;">
      </td></tr>

      <!-- Detection frame -->
      {"<tr><td style='padding:20px 32px 8px;'><p style='margin:0;font-size:13px;color:#757575;'>Detection frame with highlighted area:</p></td></tr>" if has_image else ""}
      {img_block}

      <!-- Footer -->
      <tr>
        <td style="background:#FAFAFA;padding:18px 32px;border-top:1px solid #EEEEEE;">
          <p style="margin:0;font-size:12px;color:#BDBDBD;text-align:center;">
            This is an automated alert from JANA Vision AI Monitoring.
            Job reference: {job_id}
          </p>
        </td>
      </tr>

    </table>
  </td></tr>
</table>
</body>
</html>"""


def _build_plain(
    display_name: str,
    alert_type: str,
    camera_name: Optional[str],
    job_id: str,
    alert_id: int,
    timestamp: str,
) -> str:
    label = _humanize(alert_type)
    cam   = camera_name or "Unknown Camera"
    return (
        f"JANA Vision AI Monitoring\n"
        f"{'=' * 40}\n\n"
        f"Alert:    {label}\n"
        f"KPI:      {display_name}\n"
        f"Camera:   {cam}\n"
        f"Time:     {timestamp}\n"
        f"Alert ID: #{alert_id}\n"
        f"Job:      {job_id}\n"
    )


def _build_message(
    cfg: dict,
    subject: str,
    plain: str,
    html: str,
    frame_bytes: Optional[bytes] = None,
) -> MIMEMultipart:
    from_addr = (
        f"{cfg['from_name']} <{cfg['from_address']}>"
        if cfg.get("from_name") else cfg["from_address"]
    )
    recipients = ", ".join(cfg["recipients"])

    if frame_bytes:
        # multipart/related wraps alternative + inline image
        outer = MIMEMultipart("related")
        outer["Subject"] = subject
        outer["From"]    = from_addr
        outer["To"]      = recipients

        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain, "plain"))
        alt.attach(MIMEText(html,  "html"))
        outer.attach(alt)

        img = MIMEImage(frame_bytes, "jpeg")
        img.add_header("Content-ID", "<alert_frame>")
        img.add_header("Content-Disposition", "inline", filename="detection.jpg")
        outer.attach(img)
        return outer
    else:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = from_addr
        msg["To"]      = recipients
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html,  "html"))
        return msg


def _send(cfg: dict, msg: MIMEMultipart) -> None:
    password = decrypt_secret(cfg["smtp_password_encrypted"]) if cfg["smtp_password_encrypted"] else ""
    context  = ssl.create_default_context()

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
    frame_bytes: Optional[bytes] = None,
) -> None:
    """Send an email for a newly-saved alert, if configured. Never raises -- failures are logged only."""
    cfg = get_email_config()
    if not cfg["enabled"] or not cfg["recipients"] or not cfg["smtp_host"]:
        return

    job         = db.get_job(job_id)
    camera_id   = job.camera_id   if job else None
    camera_name = job.camera_name if job else None

    label     = _humanize(alert_type)
    cam_label = camera_name or "Unknown Camera"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subject   = f"[Vision AI] {label} — {cam_label}"

    plain = _build_plain(display_name, alert_type, camera_name, job_id, alert_id, timestamp)
    html  = _build_html(display_name, alert_type, camera_name, job_id, alert_id, timestamp,
                        has_image=bool(frame_bytes))
    msg   = _build_message(cfg, subject, plain, html, frame_bytes)

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
    """Synchronous send used by the admin UI's test button — raises on failure."""
    cfg       = cfg or get_email_config()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subject   = "[Vision AI] Test email — JANA Vision AI Monitoring"
    plain = (
        "JANA Vision AI Monitoring\n"
        "This is a test email. If you received it, SMTP is configured correctly.\n"
        f"Sent at: {timestamp}"
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#F5F5F5;font-family:Arial,Helvetica,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#F5F5F5;padding:32px 0;">
  <tr><td align="center">
    <table width="620" cellpadding="0" cellspacing="0"
           style="background:#FFFFFF;border-radius:12px;overflow:hidden;
                  box-shadow:0 2px 8px rgba(0,0,0,0.08);">
      <tr>
        <td style="background:#1A237E;padding:28px 32px;">
          <p style="margin:0;color:rgba(255,255,255,0.75);font-size:12px;
                    letter-spacing:1px;text-transform:uppercase;">JANA Vision AI Monitoring</p>
          <h1 style="margin:6px 0 0;color:#FFFFFF;font-size:22px;font-weight:700;">
            Email Configuration Test</h1>
        </td>
      </tr>
      <tr>
        <td style="padding:32px;">
          <p style="margin:0;font-size:16px;color:#212121;">
            Your email alert configuration is working correctly.</p>
          <p style="margin:16px 0 0;font-size:14px;color:#757575;">
            You will receive alerts like this whenever Vision AI detects an event on your cameras.</p>
          <p style="margin:16px 0 0;font-size:13px;color:#BDBDBD;">Sent at: {timestamp}</p>
        </td>
      </tr>
      <tr>
        <td style="background:#FAFAFA;padding:18px 32px;border-top:1px solid #EEEEEE;">
          <p style="margin:0;font-size:12px;color:#BDBDBD;text-align:center;">
            This is an automated message from JANA Vision AI Monitoring.</p>
        </td>
      </tr>
    </table>
  </td></tr>
</table>
</body>
</html>"""

    msg = _build_message(cfg, subject, plain, html)
    try:
        _send(cfg, msg)
        db.create_email_log(status="sent", subject=subject, recipients=cfg["recipients"], alert_type="test")
    except Exception as exc:
        db.create_email_log(status="failed", subject=subject, recipients=cfg["recipients"], alert_type="test", error=str(exc))
        raise
