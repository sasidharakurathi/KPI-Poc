"""KPI model catalog endpoints - Phase 1.

Implements:
  GET    /api/config/kpi-models
  POST   /api/config/kpi-models
  GET    /api/config/kpi-models/{id}
  DELETE /api/config/kpi-models/{id}   (only when no KPI references it)
  PATCH  /api/config/kpi-models/{id}/toggle   Enable / disable

No Edit action by design (PRD 3.5): disable the wrong one, create a new entry.

Shared across every organization on this deployment, not org-scoped — see
app.db.models.domain_config.KpiModelCatalog's docstring. Every endpoint here
still requires authentication + the "configuration" permission, just not
ownership of a specific org's row (there isn't one).

Models used: KpiModelCatalog (app.db.models.domain_config)
Schemas: KpiModelCreate, KpiModelResponse (app.schemas.config)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from app.core.dependencies import DbSession, require_permission
from app.db.models.domain_config import KpiModelCatalog
from app.schemas.config import KpiModelCreate, KpiModelResponse

router = APIRouter(prefix="/api/config/kpi-models", tags=["config-kpi-models"])


@router.get("", response_model=list[KpiModelResponse], summary="List KPI Models")
async def list_kpi_models(
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "view")),
):
    """List every catalog entry, enabled or disabled."""
    models = session.exec(select(KpiModelCatalog).order_by(KpiModelCatalog.id)).all()
    return models


@router.post("", response_model=KpiModelResponse, status_code=201, summary="Create KPI Model")
async def create_kpi_model(
    model_in: KpiModelCreate,
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "create")),
):
    existing = session.exec(
        select(KpiModelCatalog).where(KpiModelCatalog.name == model_in.name)
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"KPI model with name '{model_in.name}' already exists.")

    model = KpiModelCatalog(**model_in.model_dump(exclude_unset=True))
    session.add(model)
    try:
        session.commit()
        session.refresh(model)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail=f"KPI model with name '{model_in.name}' already exists.")

    return model


@router.get("/{id}", response_model=KpiModelResponse, summary="Get KPI Model")
async def get_kpi_model(
    id: int,
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "view")),
):
    model = session.get(KpiModelCatalog, id)
    if not model:
        raise HTTPException(status_code=404, detail="KPI model not found.")
    return model


@router.delete("/{id}", status_code=200, summary="Delete KPI Model")
async def delete_kpi_model(
    id: int,
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "delete")),
):
    model = session.get(KpiModelCatalog, id)
    if not model:
        raise HTTPException(status_code=404, detail="KPI model not found.")

    # PRD §3.5: "Delete is only available when no KPI currently references
    # the model." There is no KPI-capability table yet to check against —
    # that's Phase 3 (KPI Management), which doesn't exist in this codebase
    # yet. Once it does, add the same reference-check pattern used by
    # priorities.py/zones.py's delete handlers here. Not a bug to "fix" now
    # — genuinely blocked on Phase 3 landing first.

    session.delete(model)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail="Database constraint violation while deleting KPI model.")

    return {"message": "row deleted successfully"}


@router.patch("/{id}/toggle", response_model=KpiModelResponse, summary="Toggle KPI Model")
async def toggle_kpi_model(
    id: int,
    session: DbSession,
    user: dict = Depends(require_permission("configuration", "edit")),
):
    model = session.get(KpiModelCatalog, id)
    if not model:
        raise HTTPException(status_code=404, detail="KPI model not found.")

    model.enabled = not model.enabled
    session.add(model)
    try:
        session.commit()
        session.refresh(model)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=400, detail="Database constraint violation while toggling KPI model.")

    return model
