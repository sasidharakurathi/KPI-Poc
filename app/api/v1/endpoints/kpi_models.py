"""KPI model catalog endpoints - Phase 1.

Implements:
  GET    /api/config/kpi-models
  POST   /api/config/kpi-models
  GET    /api/config/kpi-models/{id}
  DELETE /api/config/kpi-models/{id}   (only when no KPI references it)
  PATCH  /api/config/kpi-models/{id}/toggle   Enable / disable

No Edit action by design (PRD 3.5): disable the wrong one, create a new entry.

Models used: KpiModelCatalog (app.db.models.domain_config)
Schemas: KpiModelCreate, KpiModelResponse (app.schemas.config)
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import select, Session
from sqlalchemy.exc import IntegrityError

from app.db import get_session
from app.db.models.domain_config import KpiModelCatalog
from app.schemas.config import KpiModelCreate, KpiModelResponse

router = APIRouter(prefix="/api/config/kpi-models", tags=["config-kpi-models"])

@router.get("", response_model=list[KpiModelResponse], summary="List KPI Models")
async def list_kpi_models(session: Session = Depends(get_session)):
    models = session.exec(select(KpiModelCatalog).where(KpiModelCatalog.enabled == True).order_by(KpiModelCatalog.id)).all()
    return models

@router.post("", response_model=KpiModelResponse, status_code=201, summary="Create KPI Model")
async def create_kpi_model(model_in: KpiModelCreate, session: Session = Depends(get_session)):
    existing = session.exec(select(KpiModelCatalog).where(KpiModelCatalog.name == model_in.name)).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"KPI model with name '{model_in.name}' already exists.")
        
    model = KpiModelCatalog(**model_in.model_dump(exclude_unset=True))
    session.add(model)
    try:
        session.commit()
        session.refresh(model)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=400, detail="Database constraint violation while creating KPI model.")
        
    return model

@router.get("/{id}", response_model=KpiModelResponse, summary="Get KPI Model")
async def get_kpi_model(id: int, session: Session = Depends(get_session)):
    model = session.get(KpiModelCatalog, id)
    if not model:
        raise HTTPException(status_code=404, detail="KPI model not found.")
    return model

@router.delete("/{id}", status_code=200, summary="Delete KPI Model")
async def delete_kpi_model(id: int, session: Session = Depends(get_session)):
    model = session.get(KpiModelCatalog, id)
    if not model:
        raise HTTPException(status_code=404, detail="KPI model not found.")
        
    # TODO: Add check here to block deletion if model is referenced by KPIs
    
    session.delete(model)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail="Database constraint violation while deleting KPI model.")
        
    return {"message": "row deleted successfully"}

@router.patch("/{id}/toggle", response_model=KpiModelResponse, summary="Toggle KPI Model")
async def toggle_kpi_model(id: int, session: Session = Depends(get_session)):
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
