"""Central email-sending service for account/transactional email - the ONLY
place activation, password-reset, and user-onboarding emails are sent from.

Every one of those always sends via the organization's default
(is_default=True, enabled=True) EmailServer row (app.db.models.domain_config)
- never the legacy app.notifications Configuration-table mechanism, which
stays reserved for KPI detection alert emails only (a separate, pipeline-tied
system; see that module's docstring).

Callers that need an email to go out as part of a user-initiated action
(inviting a user, requesting a password reset) MUST call
get_default_email_server() first and let its 422 propagate - silently
continuing without telling the caller email failed is exactly the kind of
edge case this module exists to close off. The one exception is org
registration itself, which cannot require an EmailServer to already exist
(nothing does, before the org does) - it uses try_get_default_email_server()
and degrades gracefully; see app.services.auth_service.
"""
import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.db.models import EmailServer
from app.email_crypto import decrypt_secret

logger = logging.getLogger(__name__)

_NO_DEFAULT_SERVER_MESSAGE = (
    "No default email server is configured for this organization. Add one "
    "under Configuration > Email Servers first."
)


def try_get_default_email_server(db: Session, org_id: Optional[int]) -> Optional[EmailServer]:
    """Returns the org's default email server, or None if there isn't one.
    Use only where the caller must succeed regardless (org registration)."""
    return db.exec(
        select(EmailServer).where(
            EmailServer.org_id == org_id,
            EmailServer.is_default == True, 
            EmailServer.enabled == True,
        )
    ).first()


def get_default_email_server(db: Session, org_id: Optional[int]) -> EmailServer:
    """Same lookup, but raises a clear 422 instead of returning None. Every
    email-dependent action taken after an org already exists (inviting a
    user, password reset, ...) should call this and let the exception
    propagate rather than silently skip sending."""
    server = try_get_default_email_server(db, org_id)
    if server is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, _NO_DEFAULT_SERVER_MESSAGE)
    return server


def send_email(server: EmailServer, to_addresses: list[str], subject: str, html: str, plain: str) -> None:
    """Synchronous send via `server`'s own credentials. Raises on failure -
    callers decide whether that's fatal (see module docstring) or best-effort
    (org registration's activation email)."""
    password = decrypt_secret(server.password_encrypted) if server.password_encrypted else ""
    from_addr = f"{server.from_name} <{server.from_address}>" if server.from_name else server.from_address

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addresses)
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))

    context = ssl.create_default_context()
    if server.use_tls:
        with smtplib.SMTP(server.smtp_host, server.smtp_port, timeout=10) as smtp:
            smtp.starttls(context=context)
            if server.username:
                smtp.login(server.username, password)
            smtp.sendmail(server.from_address, to_addresses, msg.as_string())
    else:
        with smtplib.SMTP_SSL(server.smtp_host, server.smtp_port, timeout=10, context=context) as smtp:
            if server.username:
                smtp.login(server.username, password)
            smtp.sendmail(server.from_address, to_addresses, msg.as_string())
