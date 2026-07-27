from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    __tablename__ = "users"  

    id: Optional[int] = Field(default=None, primary_key=True)
    employee_id: Optional[str] = None
    full_name: str                                
    designation: Optional[str] = None
    personal_email: Optional[str] = Field(default=None, unique=True)
    phone: Optional[str] = None                  
    address: Optional[str] = None
    username: str = Field(unique=True, index=True)  
    login_email: str = Field(unique=True)
    password_hash: str
    org_id: Optional[int] = Field(default=None, foreign_key="organizations.id")
    role_id: Optional[int] = Field(default=None, foreign_key="roles.id")
    status: str = Field(default="active")
    
    verify_status: bool = Field(default=False)
    force_password_change: bool = Field(default=False)
    failed_login_attempts: int = Field(default=0)
    locked_until: Optional[datetime] = None
    
    reset_token: Optional[str] = None
    reset_token_expires: Optional[datetime] = None
    last_login_at: Optional[datetime] = None
    mfa_enabled: bool = Field(default=False)  
    
    token_version: int = Field(default=0)
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None


class RefreshToken(SQLModel, table=True):
    """Server-tracked refresh tokens for JWT session management."""
    __tablename__ = "refresh_tokens"  

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)                
    token_hash: str = Field(unique=True)            
    expires_at: datetime
    revoked: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
