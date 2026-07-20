from datetime import datetime
from typing import Optional

from sqlmodel import Column, Field, SQLModel
from sqlalchemy import JSON as _JSON


class AuditLog(SQLModel, table=True):
    """Immutable record of every create/update/delete/enable/disable action."""
    __tablename__ = "audit_logs"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    entity_type: str        # "camera" | "kpi_model" | "user" | "role" — matches the frontend's AuditEntity exactly
    entity_id: Optional[str] = None
    entity_name: Optional[str] = None
    action: str             # "create" | "update" | "delete" | "enable" | "disable"
    summary: Optional[str] = None   # human-readable one-liner, e.g. 'Created camera "CAM-01".'
    before: Optional[dict] = Field(default=None, sa_column=Column(_JSON))
    after: Optional[dict] = Field(default=None, sa_column=Column(_JSON))
    actor_id: Optional[int] = None
    actor_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
