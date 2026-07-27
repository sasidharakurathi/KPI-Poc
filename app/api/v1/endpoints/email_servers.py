"""Email server (SMTP) configuration endpoints - Phase 1.

Implements:
  GET    /api/config/email-servers
  POST   /api/config/email-servers
  GET    /api/config/email-servers/{id}
  PUT    /api/config/email-servers/{id}
  DELETE /api/config/email-servers/{id}   (only when no Role references it)
  POST   /api/config/email-servers/{id}/test   Send a test email

Password is encrypted using app.email_crypto before storage and never
returned. org_id is always derived from the authenticated caller
(require_permission), never from the request body. Setting is_default=true
un-defaults every other server in the same org, so "the org's default
server" is always unambiguous - matches the PRD's "one implicitly treated as
the organization default."

Models used: EmailServer (app.db.models.domain_config)
Schemas: EmailServerCreate, EmailServerUpdate, EmailServerResponse (app.schemas.config)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from app.core.dependencies import DbSession, require_permission
from app.db.models.domain_config import EmailServer
from app.schemas.config import EmailServerCreate, EmailServerResponse, EmailServerUpdate
from app.email_crypto import encrypt_secret

router = APIRouter(prefix="/api/config/email-servers", tags=["config-email-servers"])


def _clear_other_defaults(session, org_id, excluding_id=None) -> None:
    others = session.exec(
        select(EmailServer).where(EmailServer.org_id == org_id, EmailServer.is_default == True)
    ).all()
    for server in others:
        if server.id != excluding_id:
            server.is_default = False
            session.add(server)


@router.get("", response_model=list[EmailServerResponse], summary="List Email Servers")
async def list_email_servers(
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "view")),
):
    """List every email server for the caller's org, enabled or disabled."""
    servers = session.exec(
        select(EmailServer).where(EmailServer.org_id == user.get("org_id")).order_by(EmailServer.id)
    ).all()
    return servers


@router.post("", response_model=EmailServerResponse, status_code=201, summary="Create Email Server")
async def create_email_server(
    server_in: EmailServerCreate,
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "create")),
):
    org_id = user.get("org_id")
    existing = session.exec(
        select(EmailServer).where(EmailServer.org_id == org_id, EmailServer.label == server_in.label)
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Email server with label '{server_in.label}' already exists.")

    server_data = server_in.model_dump(exclude={"password"}, exclude_unset=True)
    server_data["password_encrypted"] = encrypt_secret(server_in.password)

    server = EmailServer(**server_data, org_id=org_id)

    if server.is_default:
        _clear_other_defaults(session, org_id)

    session.add(server)
    try:
        session.commit()
        session.refresh(server)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail=f"Email server with label '{server_in.label}' already exists.")

    return server


@router.get("/{id}", response_model=EmailServerResponse, summary="Get Email Server")
async def get_email_server(
    id: int,
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "view")),
):
    server = session.get(EmailServer, id)
    if not server or server.org_id != user.get("org_id"):
        raise HTTPException(status_code=404, detail="Email server not found.")
    return server


@router.put("/{id}", response_model=EmailServerResponse, summary="Update Email Server")
async def update_email_server(
    id: int,
    server_in: EmailServerUpdate,
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "edit")),
):
    org_id = user.get("org_id")
    server = session.get(EmailServer, id)
    if not server or server.org_id != org_id:
        raise HTTPException(status_code=404, detail="Email server not found.")

    update_data = server_in.model_dump(exclude={"password"}, exclude_unset=True)
    if server_in.password is not None:
        update_data["password_encrypted"] = encrypt_secret(server_in.password)

    for key, value in update_data.items():
        setattr(server, key, value)

    if server.is_default:
        _clear_other_defaults(session, org_id, excluding_id=server.id)

    session.add(server)
    try:
        session.commit()
        session.refresh(server)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=400, detail="Database constraint violation while updating email server.")

    return server


@router.delete("/{id}", status_code=200, summary="Delete Email Server")
async def delete_email_server(
    id: int,
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "delete")),
):
    server = session.get(EmailServer, id)
    if not server or server.org_id != user.get("org_id"):
        raise HTTPException(status_code=404, detail="Email server not found.")

    from app.db.models.role import Role
    in_use_by_role = session.exec(select(Role).where(Role.default_email_server_id == id)).first()

    if in_use_by_role:
        raise HTTPException(status_code=409, detail="Email server is in use by one or more roles.")

    session.delete(server)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail="Database constraint violation while deleting email server.")

    return {"message": "row deleted successfully"}


from pydantic import BaseModel, EmailStr

from app.services.email_service import send_email

class TestEmailRequest(BaseModel):
    recipient: EmailStr

@router.post("/{id}/test", summary="Send Test Email")
async def send_test_email(
    id: int,
    payload: TestEmailRequest,
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "edit")),
):
    server = session.get(EmailServer, id)
    if not server or server.org_id != user.get("org_id"):
        raise HTTPException(status_code=404, detail="Email server not found.")

    plain = "This is a test email sent from the Vision AI system to verify your SMTP configuration."
    html = f"<p>{plain}</p>"
    try:
        send_email(server, [payload.recipient], "Test Email from Vision AI", html, plain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send test email: {str(e)}")

    return {"message": f"Test email sent successfully to {payload.recipient}"}
