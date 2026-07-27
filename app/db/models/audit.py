from datetime import datetime
from typing import Optional

from sqlmodel import Column, Field, SQLModel
from sqlalchemy import JSON as _JSON


class AuditLog(SQLModel, table=True):
    """Immutable record of every create/update/delete/enable/disable action."""
    __tablename__ = "audit_logs" 

    id: Optional[int] = Field(default=None, primary_key=True)
    entity_type: str
    entity_id: Optional[str] = None
    entity_name: Optional[str] = None
    action: str           
    summary: Optional[str] = None   
    before: Optional[dict] = Field(default=None, sa_column=Column(_JSON))
    after: Optional[dict] = Field(default=None, sa_column=Column(_JSON))
    actor_id: Optional[int] = None
    actor_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
