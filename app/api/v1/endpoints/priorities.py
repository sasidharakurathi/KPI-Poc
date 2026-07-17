"""Priority configuration endpoints — Phase 1.

Implements:
  GET    /api/config/priorities           List all priorities
  POST   /api/config/priorities           Create a priority
  PUT    /api/config/priorities/{id}      Update name/color
  DELETE /api/config/priorities/{id}      Delete (only if no cameras or KPIs reference it)
  PATCH  /api/config/priorities/{id}/toggle   Enable / disable

Models used: Priority (app.db.models.domain_config)
Schemas: PriorityCreate, PriorityResponse (app.schemas.config)
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import select, Session
from sqlalchemy.exc import IntegrityError

from app.db import get_session
from app.db.models.domain_config import Priority
from app.schemas.config import PriorityCreate, PriorityResponse, PriorityUpdate

router = APIRouter(prefix="/api/config/priorities", tags=["config-priorities"])

@router.get("", response_model=list[PriorityResponse], summary="List Priorities")
async def list_priorities(session: Session = Depends(get_session)):
    """List all enabled priorities."""
    priorities = session.exec(select(Priority).where(Priority.enabled == True).order_by(Priority.id)).all()
    return priorities

@router.get("", response_model=list[PriorityResponse], summary="List Priorities")
async def list_priorities(session: Session = Depends(get_session)):
    """List all enabled priorities."""
    priorities = session.exec(select(Priority).where(Priority.enabled == True).order_by(Priority.id)).all()
    return priorities


@router.post("", response_model=PriorityResponse, status_code=201, summary="Create Priority")
async def create_priority(priority_in: PriorityCreate, session: Session = Depends(get_session)):
    existing = session.exec(select(Priority).where(Priority.name == priority_in.name)).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Priority with name '{priority_in.name}' already exists.")
    
    priority_data = priority_in.model_dump(exclude_unset=True)
    priority = Priority(**priority_data)
    
    session.add(priority)
    try:
        session.commit()
        session.refresh(priority)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=400, detail="Database constraint violation while creating priority.")
        
    return priority

@router.put("/{id}", response_model=PriorityResponse, summary="Update Priority")
async def update_priority(id: int, priority_in: PriorityUpdate, session: Session = Depends(get_session)):
    priority = session.get(Priority, id)
    if not priority:
        raise HTTPException(status_code=404, detail="Priority not found.")
    
    if priority_in.name is not None and priority_in.name != priority.name:
        existing = session.exec(select(Priority).where(Priority.name == priority_in.name)).first()
        if existing:
            raise HTTPException(status_code=409, detail=f"Priority with name '{priority_in.name}' already exists.")
            
    update_data = priority_in.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(priority, key, value)
        
    session.add(priority)
    try:
        session.commit()
        session.refresh(priority)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=400, detail="Database constraint violation while updating priority.")
        
    return priority

@router.delete("/{id}", status_code=200, summary="Delete Priority")
async def delete_priority(id: int, session: Session = Depends(get_session)):
    priority = session.get(Priority, id)
    if not priority:
        raise HTTPException(status_code=404, detail="Priority not found.")
        
    from app.db.models.camera import Camera
    in_use_by_camera = session.exec(
        select(Camera).where((Camera.priority_id == id) | (Camera.priority == priority.name))
    ).first()
    
    if in_use_by_camera:
        raise HTTPException(status_code=409, detail="Priority is in use by one or more cameras or KPIs.")
        
    session.delete(priority)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail="Database constraint violation while deleting priority.")
        
    return {"message": "row deleted successfully"}

@router.patch("/{id}/toggle", response_model=PriorityResponse, summary="Toggle Priority")
async def toggle_priority(id: int, session: Session = Depends(get_session)):
    priority = session.get(Priority, id)
    if not priority:
        raise HTTPException(status_code=404, detail="Priority not found.")
        
    priority.enabled = not priority.enabled
    session.add(priority)
    try:
        session.commit()
        session.refresh(priority)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=400, detail="Database constraint violation while toggling priority.")
        
    return priority
