from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel
from sqlalchemy import Column, BigInteger, String, Text, Boolean, TIMESTAMP, text

class KPIConfiguration(SQLModel, table=True):
    __tablename__ = "kpi_configuration"

    id: Optional[int] = Field(default=None, sa_column=Column(BigInteger, primary_key=True, autoincrement=True))
    kpi_name: str = Field(sa_column=Column(String(255), nullable=False))
    assigned_models: Optional[str] = Field(default=None, sa_column=Column(Text))
    parameters: Optional[str] = Field(default=None, sa_column=Column(Text))
    enable_status: Optional[bool] = Field(default=True, sa_column=Column(Boolean, server_default=text("TRUE")))
    created_at: Optional[datetime] = Field(default=None, sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP")))
    created_by: Optional[str] = Field(default=None, sa_column=Column(String(100)))
    updated_at: Optional[datetime] = Field(default=None, sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP")))
    updated_by: Optional[str] = Field(default=None, sa_column=Column(String(100)))
