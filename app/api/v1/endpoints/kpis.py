from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException

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

def _registered_class_or_404(name: str):
    cls = get_registry().get(name)
    if not cls:
        raise HTTPException(status_code=404, detail=f"KPI '{name}' is not a registered detector.")
    return cls


def _validate_model_ids(session: DbSession, model_ids: list[str]) -> list[str]:
    for raw in model_ids:
        try:
            int(raw)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail=f"Invalid model id: {raw!r}")
    return model_ids


def _to_catalog_response(config: KPIConfiguration) -> KpiCatalogResponse:
    cls = get_registry().get(config.kpi_name)
    return KpiCatalogResponse(
        id=str(config.id),
        key=config.kpi_name,
        display_name=cls.display_name if cls else config.kpi_name,
        description=config.description or "",
        category=config.category or "operations",
        enabled=config.enable_status if config.enable_status is not None else True,
        config=config.parameters or {},
        model_ids=[str(m) for m in (config.assigned_models or [])],
        added_at=config.created_at.isoformat() if config.created_at else datetime.utcnow().isoformat(),
    )


@router.get("/catalog", response_model=list[KpiCatalogResponse])
async def list_kpi_catalog(
    session: DbSession,
    user: dict = Depends(require_permission("kpi_management", "view")),
):
    configs = session.exec(select(KPIConfiguration)).all()
    return [_to_catalog_response(c) for c in configs]


@router.get("/catalog/{name}", response_model=KpiCatalogResponse)
async def get_kpi_catalog(
    name: str,
    session: DbSession,
    user: dict = Depends(require_permission("kpi_management", "view")),
):
    config = session.exec(
        select(KPIConfiguration).where(KPIConfiguration.kpi_name == name)
    ).first()
    if not config:
        raise HTTPException(status_code=404, detail="KPI Configuration not found.")
    return _to_catalog_response(config)


@router.put("/catalog/{name}", response_model=KpiCatalogResponse)
async def update_kpi_catalog(
    name: str,
    payload: KpiCatalogUpdate,
    session: DbSession,
    user: dict = Depends(require_permission("kpi_management", "edit")),
):
    cls = _registered_class_or_404(name)
    actor = user.get("username")

    config = session.exec(
        select(KPIConfiguration).where(KPIConfiguration.kpi_name == name)
    ).first()

    if not config:
        config = KPIConfiguration(kpi_name=name, created_by=actor)

    if payload.model_ids is not None:
        config.assigned_models = _validate_model_ids(session, payload.model_ids)
    if payload.config is not None:
        config.parameters = payload.config
    if payload.description is not None:
        config.description = payload.description
    if payload.category is not None:
        config.category = payload.category
    config.updated_by = actor
    config.updated_at = datetime.utcnow()

    session.add(config)
    session.commit()
    session.refresh(config)

    if payload.config is not None:
        update_kpi_config(cls.__name__, payload.config)

    return _to_catalog_response(config)


@router.patch("/catalog/{name}/toggle", response_model=KpiCatalogResponse)
async def toggle_kpi_catalog(
    name: str,
    session: DbSession,
    user: dict = Depends(require_permission("kpi_management", "edit")),
):
    cls = _registered_class_or_404(name)
    config = session.exec(
        select(KPIConfiguration).where(KPIConfiguration.kpi_name == name)
    ).first()
    if not config:
        raise HTTPException(status_code=404, detail="KPI Configuration not found.")

    config.enable_status = not config.enable_status
    config.updated_by = user.get("username")
    config.updated_at = datetime.utcnow()
    session.add(config)
    session.commit()
    session.refresh(config)

    update_kpi_config(cls.__name__, {"enabled": config.enable_status})

    return _to_catalog_response(config)
