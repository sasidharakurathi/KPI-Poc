"""KPI detection alert emails.

Recipients and SMTP config are both resolved live from the database for the
alert's own organization: recipients from Role.kpi_names (which users' roles
are allowed to see this KPI - see app.services.kpi_role_scope), and SMTP
config from that org's active EmailServer row (is_default=True, enabled=True
- see app.services.email_service). No hardcoded fallback of either kind -
if the org has no default EmailServer configured, sending is skipped and
logged as a failed EmailLog rather than silently using some other org's or a
built-in server.

This module owns its own MIME-building (inline detection-frame image support
that app.services.email_service's account-email sender doesn't need) but
sends through the same EmailServer credentials/connection logic.
"""
import logging
import smtplib
import ssl
import threading
from datetime import datetime, timezone
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from . import db
from .db.models import Alert, EmailServer
from .email_crypto import decrypt_secret
from .services.email_service import try_get_default_email_server
from .services.kpi_role_scope import eligible_recipients_for_alert

logger = logging.getLogger(__name__)

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
    "camera_offline":        "Camera Offline",
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
    "camera_offline":        "#424242",
}
_DEFAULT_COLOR = "#1A237E"


def _humanize(alert_type: str) -> str:
    return _ALERT_LABELS.get(alert_type, alert_type.replace("_", " ").title())


def _build_html(
    display_name: str,
    alert_type: str,
    camera_name: Optional[str],
    job_id: Optional[str],
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
                    letter-spacing:1px;text-transform:uppercase;">Vision AI Monitoring</p>
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
            This is an automated alert from Vision AI Monitoring.
            Job reference: {job_id or "N/A"}
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
    job_id: Optional[str],
    alert_id: int,
    timestamp: str,
) -> str:
    label = _humanize(alert_type)
    cam   = camera_name or "Unknown Camera"
    return (
        f"Vision AI Monitoring\n"
        f"{'=' * 40}\n\n"
        f"Alert:    {label}\n"
        f"KPI:      {display_name}\n"
        f"Camera:   {cam}\n"
        f"Time:     {timestamp}\n"
        f"Alert ID: #{alert_id}\n"
        f"Job:      {job_id or 'N/A'}\n"
    )


def _build_message(
    server: EmailServer,
    recipients: list[str],
    subject: str,
    plain: str,
    html: str,
    frame_bytes: Optional[bytes] = None,
) -> MIMEMultipart:
    from_addr = (
        f"{server.from_name} <{server.from_address}>"
        if server.from_name else server.from_address
    )
    to_header = ", ".join(recipients)

    if frame_bytes:
        # multipart/related wraps alternative + inline image
        outer = MIMEMultipart("related")
        outer["Subject"] = subject
        outer["From"]    = from_addr
        outer["To"]      = to_header

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
        msg["To"]      = to_header
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html,  "html"))
        return msg


def _send(server: EmailServer, recipients: list[str], msg: MIMEMultipart) -> None:
    password = decrypt_secret(server.password_encrypted) if server.password_encrypted else ""
    context  = ssl.create_default_context()

    if server.use_tls:
        with smtplib.SMTP(server.smtp_host, server.smtp_port, timeout=10) as smtp:
            smtp.starttls(context=context)
            if server.username:
                smtp.login(server.username, password)
            smtp.sendmail(server.from_address, recipients, msg.as_string())
    else:
        with smtplib.SMTP_SSL(server.smtp_host, server.smtp_port, timeout=10, context=context) as smtp:
            if server.username:
                smtp.login(server.username, password)
            smtp.sendmail(server.from_address, recipients, msg.as_string())


def notify_alert(
    kpi_name: str,
    display_name: str,
    alert_type: str,
    job_id: Optional[str],
    alert_id: int,
    confidence: float,
    frame_bytes: Optional[bytes] = None,
    camera_id: Optional[str] = None,
    camera_name: Optional[str] = None,
) -> None:
    """Send an email for a newly-saved alert to every user whose role can see
    this KPI (see app.services.kpi_role_scope.eligible_recipients_for_alert),
    via the alert's organization's active EmailServer. Never raises -
    failures are logged only.

    camera_id/camera_name are auto-resolved from the job when omitted (the
    normal case for video-detection alerts). Pass them explicitly for alerts
    with no job at all - e.g. app.services.camera_heartbeat's connectivity
    alerts."""
    with db.get_session_ctx() as session:
        alert_row = session.get(Alert, alert_id)
        org_id = alert_row.org_id if alert_row else None
        recipients = eligible_recipients_for_alert(session, org_id, kpi_name)
        server = try_get_default_email_server(session, org_id)

    if not recipients:
        return

    if camera_id is None and camera_name is None and job_id:
        job         = db.get_job(job_id)
        camera_id   = job.camera_id   if job else None
        camera_name = job.camera_name if job else None

    label     = _humanize(alert_type)
    cam_label = camera_name or "Unknown Camera"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subject   = f"[Vision AI] {label} - {cam_label}"

    if server is None:
        logger.warning(
            f"[email] no default email server configured for org {org_id} - "
            f"skipping notification for alert #{alert_id} ({kpi_name})"
        )
        db.create_email_log(
            status="failed", subject=subject, recipients=recipients,
            alert_id=alert_id, org_id=org_id, kpi_name=kpi_name, alert_type=alert_type,
            camera_id=camera_id, camera_name=camera_name,
            error="No default email server configured for this organization.",
        )
        return

    plain = _build_plain(display_name, alert_type, camera_name, job_id, alert_id, timestamp)
    html  = _build_html(display_name, alert_type, camera_name, job_id, alert_id, timestamp,
                        has_image=bool(frame_bytes))
    msg   = _build_message(server, recipients, subject, plain, html, frame_bytes)

    def _worker() -> None:
        try:
            _send(server, recipients, msg)
            logger.info(f"[email] sent notification for alert #{alert_id} ({kpi_name}) to {len(recipients)} recipient(s)")
            db.create_email_log(
                status="sent", subject=subject, recipients=recipients,
                alert_id=alert_id, org_id=org_id, kpi_name=kpi_name, alert_type=alert_type,
                camera_id=camera_id, camera_name=camera_name,
            )
        except Exception as exc:
            logger.exception(f"[email] failed to send notification for alert #{alert_id} ({kpi_name})")
            db.create_email_log(
                status="failed", subject=subject, recipients=recipients,
                alert_id=alert_id, org_id=org_id, kpi_name=kpi_name, alert_type=alert_type,
                camera_id=camera_id, camera_name=camera_name, error=str(exc),
            )

    threading.Thread(target=_worker, daemon=True, name="email-notify").start()
