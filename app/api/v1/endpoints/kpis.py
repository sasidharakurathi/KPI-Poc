from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException

from app.config_loader import (
    get_all as get_full_config,
    get_kpi_config,
    get_kpi_param,
    get_kpi_registry,
    reload as reload_config,
    update_kpi_config,
)
from app.kpis import get_registered_kpis, get_registry, list_registered_names
from app.schemas import KPIInfo, KPISettingsItem, KPISettingsResponse, RegisteredKPIsResponse

router = APIRouter(prefix="/api/kpis", tags=["kpis"])


@router.get("", response_model=RegisteredKPIsResponse)
async def list_kpis():
    kpis = get_registered_kpis()
    return RegisteredKPIsResponse(
        count=len(kpis),
        kpis=[KPIInfo(name=k.name, display_name=k.display_name) for k in kpis],
    )


@router.get("/settings", response_model=KPISettingsResponse)
async def list_kpi_settings():
    items = [
        KPISettingsItem(
            name=cls.name,
            display_name=cls.display_name,
            enabled=get_kpi_param(cls.__name__, "enabled", True),
            config=get_kpi_config(cls.__name__),
        )
        for cls in get_registry().values()
    ]
    return KPISettingsResponse(count=len(items), kpis=items)


@router.put("/{name}/config", response_model=KPISettingsItem)
async def update_kpi_settings(name: str, updates: Annotated[dict[str, Any], Body()]):
    cls = get_registry().get(name)
    if not cls:
        raise HTTPException(status_code=404, detail=f"KPI '{name}' not found.")
    new_cfg = update_kpi_config(cls.__name__, updates)
    return KPISettingsItem(
        name=cls.name,
        display_name=cls.display_name,
        enabled=new_cfg.get("enabled", True),
        config=new_cfg,
    )
