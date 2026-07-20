from typing import Any, Optional
from datetime import datetime

from pydantic import BaseModel


class KPIInfo(BaseModel):
    name: str
    display_name: str


class RegisteredKPIsResponse(BaseModel):
    count: int
    kpis: list[KPIInfo]


class KPISettingsItem(BaseModel):
    name: str
    display_name: str
    enabled: bool
    config: dict[str, Any]


class KPISettingsResponse(BaseModel):
    count: int
    kpis: list[KPISettingsItem]


from pydantic import BaseModel, field_validator


class KpiCatalogUpdate(BaseModel):
    assigned_models: Optional[list[str]] = None
    parameters: Optional[dict[str, Any]] = None

    @field_validator('assigned_models', mode='before')
    @classmethod
    def prevent_string_coercion(cls, v):
        if isinstance(v, str):
            raise ValueError("assigned_models must be a list of strings (e.g., [\"models/enable.py\"]), not a single string.")
        return v


class KpiCatalogResponse(BaseModel):
    id: int
    kpi_name: str
    assigned_models: Optional[list[str]] = None
    parameters: Optional[dict[str, Any]] = None
    enable_status: bool
    created_at: datetime
    created_by: Optional[str] = None
    updated_at: datetime
    updated_by: Optional[str] = None
