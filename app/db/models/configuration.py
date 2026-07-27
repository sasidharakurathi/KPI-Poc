from datetime import datetime

from sqlmodel import Field, SQLModel


class Configuration(SQLModel, table=True):
    """Generic key→JSON-string config store. Kept for backward compatibility with
    existing email/timezone settings. New modules use dedicated tables instead."""
    __tablename__ = "configurations"

    name: str = Field(primary_key=True)
    value: str = ""
    updated_at: datetime = Field(default_factory=datetime.utcnow)
