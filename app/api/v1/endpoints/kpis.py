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
from app.schemas.kpi import KpiCatalogResponse, KpiCatalogUpdate
from app.db.models.kpi_configuration import KPIConfiguration
from app.core.dependencies import DbSession, require_permission
from sqlmodel import select
from fastapi import Depends
import json

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

def _parse_kpi_config(kpi_config: KPIConfiguration) -> KpiCatalogResponse:
    data = kpi_config.model_dump()
    if data.get("assigned_models"):
        try:
            data["assigned_models"] = json.loads(data["assigned_models"])
        except json.JSONDecodeError:
            pass
    if data.get("parameters"):
        try:
            data["parameters"] = json.loads(data["parameters"])
        except json.JSONDecodeError:
            pass
    return KpiCatalogResponse(**data)

@router.get("/catalog", response_model=list[KpiCatalogResponse])
async def list_kpi_catalog(
    session: DbSession,
    # user: dict = Depends(require_permission("configuration", "view"))
):
    configs = session.exec(select(KPIConfiguration)).all()
    return [_parse_kpi_config(c) for c in configs]

@router.get("/catalog/{name}", response_model=KpiCatalogResponse)
async def get_kpi_catalog(
    name: str,
    session: DbSession,
    # user: dict = Depends(require_permission("configuration", "view"))
):
    config = session.exec(select(KPIConfiguration).where(KPIConfiguration.kpi_name == name)).first()
    if not config:
        raise HTTPException(status_code=404, detail="KPI Configuration not found.")
    return _parse_kpi_config(config)

@router.put("/catalog/{name}", response_model=KpiCatalogResponse)
async def update_kpi_catalog(
    name: str,
    payload: KpiCatalogUpdate,
    session: DbSession,
    # user: dict = Depends(require_permission("configuration", "edit"))
):
    config = session.exec(select(KPIConfiguration).where(KPIConfiguration.kpi_name == name)).first()
    
    if not config:
        config = KPIConfiguration(kpi_name=name)
        
    if payload.assigned_models is not None:
        config.assigned_models = json.dumps(payload.assigned_models) if not isinstance(payload.assigned_models, str) else payload.assigned_models
    if payload.parameters is not None:
        config.parameters = json.dumps(payload.parameters) if not isinstance(payload.parameters, str) else payload.parameters
        
    session.add(config)
    session.commit()
    session.refresh(config)
    return _parse_kpi_config(config)

@router.patch("/catalog/{name}/toggle", response_model=KpiCatalogResponse)
async def toggle_kpi_catalog(
    name: str,
    session: DbSession,
    # user: dict = Depends(require_permission("configuration", "edit"))
):
    config = session.exec(select(KPIConfiguration).where(KPIConfiguration.kpi_name == name)).first()
    if not config:
        raise HTTPException(status_code=404, detail="KPI Configuration not found.")
        
    config.enable_status = not config.enable_status
    session.add(config)
    session.commit()
    session.refresh(config)
    return _parse_kpi_config(config)
