from datetime import datetime
from typing import Optional

from sqlmodel import Column, Field, SQLModel, UniqueConstraint
from sqlalchemy import JSON as _JSON


class Role(SQLModel, table=True):
    __tablename__ = "roles"  
    
    __table_args__ = (UniqueConstraint("org_id", "name", name="uq_roles_org_id_name"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str                               
    description: Optional[str] = None
    permissions: Optional[dict] = Field(default=None, sa_column=Column(_JSON))
    
    default_email_server_id: Optional[int] = Field(default=None, foreign_key="email_servers.id")
    
    zone_ids: list = Field(default_factory=list, sa_column=Column(_JSON))
    kpi_names: list = Field(default_factory=list, sa_column=Column(_JSON))
    is_system: bool = Field(default=False)
    org_id: Optional[int] = Field(default=None, foreign_key="organizations.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None
