"""Reference-data tables: Priority, Timezone, Zone, EmailServer, KpiModelCatalog.
These are the Phase 1 Configuration module tables from the PRD.

Priority.name / EmailServer.label are unique per organization, not globally
- enforced via a composite (org_id, name) unique constraint rather than a
single-column one, since this platform hosts multiple organizations and two
different orgs must each be free to use "Critical" as a priority name,
"Primary SMTP" as an email server label, etc.

KpiModelCatalog is the one exception: it's a single shared catalog of
detection-model files across every organization on the deployment (there's
one physical model file per KPI regardless of tenant count), so its name is
globally unique. org_id is left on the column - additive-only migration
policy - but is no longer read or written by any endpoint.
"""
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel, UniqueConstraint


class Priority(SQLModel, table=True):
    __tablename__ = "priorities"  
    __table_args__ = (UniqueConstraint("org_id", "name", name="uq_priorities_org_id_name"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str                                
    color: str                               
    level: int = Field(default=99)           
    enabled: bool = Field(default=True)
    org_id: Optional[int] = Field(default=None, foreign_key="organizations.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None


class Timezone(SQLModel, table=True):
    __tablename__ = "timezones"  

    id: Optional[int] = Field(default=None, primary_key=True)
    abbreviation: str
    timezone_name: str
    utc_offset: Optional[str] = None
    gmt_offset: Optional[str] = None
    enabled: bool = Field(default=True)


class Zone(SQLModel, table=True):
    __tablename__ = "zones"  

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str                                   
    org_id: Optional[int] = Field(default=None, foreign_key="organizations.id")
    
    
    
    timezone: Optional[str] = None
    timezone_id: Optional[int] = Field(default=None, foreign_key="timezones.id")
    description: Optional[str] = None
    enabled: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None


class EmailServer(SQLModel, table=True):
    __tablename__ = "email_servers"  
    __table_args__ = (UniqueConstraint("org_id", "label", name="uq_email_servers_org_id_label"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    label: str                                  
    smtp_host: str
    smtp_port: int = 587
    username: str
    password_encrypted: str                     
    use_tls: bool = Field(default=True)
    from_address: str
    from_name: str
    enabled: bool = Field(default=True)
    is_default: bool = Field(default=False)     
    org_id: Optional[int] = Field(default=None, foreign_key="organizations.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None


class KpiModelCatalog(SQLModel, table=True):
    """Catalog of AI model files, shared across every organization on this
    deployment - not org-scoped (see module docstring)."""
    __tablename__ = "kpi_model_catalog"  

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True)              
    model_path: str                             
    confidence_threshold: float = 0.5          
    enabled: bool = Field(default=True)
    
    
    org_id: Optional[int] = Field(default=None, foreign_key="organizations.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None
