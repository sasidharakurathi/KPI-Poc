from typing import Any, Optional

from pydantic import BaseModel, field_validator


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


KPI_CATEGORIES = ("safety", "compliance", "operations", "security")


class KpiCatalogUpdate(BaseModel):
    """Field names/values match the frontend's real contract exactly
    (vision-ai-frontend/src/api/kpiModels.ts KpiModelInput + the
    setEnabled/updateConfig/updateModels mutations) - model_ids references
    Phase 1's KpiModelCatalog (Configuration > KPI Models), not this
    catalog's own rows and not the code-level detector registry."""
    model_ids: Optional[list[str]] = None
    config: Optional[dict[str, Any]] = None
    description: Optional[str] = None
    category: Optional[str] = None

    @field_validator("model_ids", mode="before")
    @classmethod
    def prevent_string_coercion(cls, v):
        if isinstance(v, str):
            raise ValueError('model_ids must be a list of strings (e.g., ["3"]), not a single string.')
        return v

    @field_validator("category")
    @classmethod
    def _valid_category(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in KPI_CATEGORIES:
            raise ValueError(f"category must be one of {KPI_CATEGORIES}.")
        return v


class KpiCatalogResponse(BaseModel):
    """Matches the frontend's KpiModelDef exactly. display_name is resolved
    live from the real detector registry (app.kpis.registry) rather than
    stored - kpi_name is gated to a real registered detector at write time,
    so this is always resolvable - matching the "resolve live, don't
    snapshot" pattern already used for zone_name/priority_name elsewhere."""
    id: str
    key: str
    display_name: str
    description: str
    category: str
    enabled: bool
    config: dict[str, Any]
    model_ids: list[str]
    added_at: str
