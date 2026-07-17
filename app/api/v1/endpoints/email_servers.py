"""Email server (SMTP) configuration endpoints — Phase 1.

Implements:
  GET    /api/config/email-servers
  POST   /api/config/email-servers
  GET    /api/config/email-servers/{id}
  PUT    /api/config/email-servers/{id}
  DELETE /api/config/email-servers/{id}   (only when no Role references it)
  POST   /api/config/email-servers/{id}/test   Send a test email

Password is encrypted using app.email_crypto before storage and never returned.

Models used: EmailServer (app.db.models.domain_config)
Schemas: EmailServerCreate, EmailServerUpdate, EmailServerResponse (app.schemas.config)
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import select, Session
from sqlalchemy.exc import IntegrityError

from app.db import get_session
from app.db.models.domain_config import EmailServer
from app.schemas.config import EmailServerCreate, EmailServerResponse, EmailServerUpdate
from app.email_crypto import encrypt_secret

router = APIRouter(prefix="/api/config/email-servers", tags=["config-email-servers"])

@router.get("", response_model=list[EmailServerResponse], summary="List Email Servers")
async def list_email_servers(session: Session = Depends(get_session)):
    """List all enabled email servers."""
    servers = session.exec(select(EmailServer).where(EmailServer.enabled == True).order_by(EmailServer.id)).all()
    return servers

@router.post("", response_model=EmailServerResponse, status_code=201, summary="Create Email Server")
async def create_email_server(server_in: EmailServerCreate, session: Session = Depends(get_session)):
    existing = session.exec(select(EmailServer).where(EmailServer.label == server_in.label)).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Email server with label '{server_in.label}' already exists.")
    
    server_data = server_in.model_dump(exclude={"password"}, exclude_unset=True)
    server_data["password_encrypted"] = encrypt_secret(server_in.password)
    
    server = EmailServer(**server_data)
    
    session.add(server)
    try:
        session.commit()
        session.refresh(server)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=400, detail="Database constraint violation while creating email server.")
        
    return server

@router.get("/{id}", response_model=EmailServerResponse, summary="Get Email Server")
async def get_email_server(id: int, session: Session = Depends(get_session)):
    server = session.get(EmailServer, id)
    if not server:
        raise HTTPException(status_code=404, detail="Email server not found.")
    return server

@router.put("/{id}", response_model=EmailServerResponse, summary="Update Email Server")
async def update_email_server(id: int, server_in: EmailServerUpdate, session: Session = Depends(get_session)):
    server = session.get(EmailServer, id)
    if not server:
        raise HTTPException(status_code=404, detail="Email server not found.")
        
    update_data = server_in.model_dump(exclude={"password"}, exclude_unset=True)
    if server_in.password is not None:
        update_data["password_encrypted"] = encrypt_secret(server_in.password)
        
    for key, value in update_data.items():
        setattr(server, key, value)
        
    session.add(server)
    try:
        session.commit()
        session.refresh(server)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=400, detail="Database constraint violation while updating email server.")
        
    return server

@router.delete("/{id}", status_code=200, summary="Delete Email Server")
async def delete_email_server(id: int, session: Session = Depends(get_session)):
    server = session.get(EmailServer, id)
    if not server:
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
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.email_crypto import decrypt_secret

class TestEmailRequest(BaseModel):
    recipient: EmailStr

@router.post("/{id}/test", summary="Send Test Email")
async def send_test_email(id: int, payload: TestEmailRequest, session: Session = Depends(get_session)):
    server = session.get(EmailServer, id)
    if not server:
        raise HTTPException(status_code=404, detail="Email server not found.")
        
    password = decrypt_secret(server.password_encrypted) if server.password_encrypted else ""
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Test Email from JANA Vision AI"
    msg["From"] = f"{server.from_name} <{server.from_address}>" if server.from_name else server.from_address
    msg["To"] = payload.recipient
    
    text = "This is a test email sent from the JANA Vision AI system to verify your SMTP configuration."
    msg.attach(MIMEText(text, "plain"))
    
    context = ssl.create_default_context()
    try:
        if server.use_tls:
            with smtplib.SMTP(server.smtp_host, server.smtp_port, timeout=10) as smtp:
                smtp.starttls(context=context)
                if server.username:
                    smtp.login(server.username, password)
                smtp.sendmail(server.from_address, [payload.recipient], msg.as_string())
        else:
            with smtplib.SMTP_SSL(server.smtp_host, server.smtp_port, timeout=10, context=context) as smtp:
                if server.username:
                    smtp.login(server.username, password)
                smtp.sendmail(server.from_address, [payload.recipient], msg.as_string())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send test email: {str(e)}")
        
    return {"message": f"Test email sent successfully to {payload.recipient}"}
