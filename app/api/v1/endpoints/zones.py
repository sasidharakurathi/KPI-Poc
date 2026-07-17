"""Zone configuration endpoints — Phase 1.

Implements:
  GET    /api/config/zones
  POST   /api/config/zones
  PUT    /api/config/zones/{id}
  DELETE /api/config/zones/{id}   (only when no cameras reference it)
  PATCH  /api/config/zones/{id}/toggle

Models used: Zone (app.db.models.domain_config)
Schemas: ZoneCreate, ZoneResponse (app.schemas.config)
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import select, Session
from sqlalchemy.exc import IntegrityError

from app.db import get_session
from app.db.models.domain_config import Zone
from app.schemas.config import ZoneCreate, ZoneResponse, ZoneUpdate

router = APIRouter(prefix="/api/config/zones", tags=["config-zones"])

@router.get("", response_model=list[ZoneResponse], summary="List Zones")
async def list_zones(session: Session = Depends(get_session)):
    """List all enabled zones."""
    zones = session.exec(select(Zone).where(Zone.enabled == True).order_by(Zone.id)).all()
    return zones

@router.post("", response_model=ZoneResponse, status_code=201, summary="Create Zone")
async def create_zone(zone_in: ZoneCreate, session: Session = Depends(get_session)):
    existing = session.exec(select(Zone).where(Zone.name == zone_in.name)).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Zone with name '{zone_in.name}' already exists.")
    
    zone_data = zone_in.model_dump(exclude_unset=True)
    zone = Zone(**zone_data)
    
    session.add(zone)
    try:
        session.commit()
        session.refresh(zone)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=400, detail="Database constraint violation while creating zone.")
        
    return zone

@router.put("/{id}", response_model=ZoneResponse, summary="Update Zone")
async def update_zone(id: int, zone_in: ZoneUpdate, session: Session = Depends(get_session)):
    zone = session.get(Zone, id)
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found.")
    
    if zone_in.name is not None and zone_in.name != zone.name:
        existing = session.exec(select(Zone).where(Zone.name == zone_in.name)).first()
        if existing:
            raise HTTPException(status_code=409, detail=f"Zone with name '{zone_in.name}' already exists.")
            
    update_data = zone_in.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(zone, key, value)
        
    session.add(zone)
    try:
        session.commit()
        session.refresh(zone)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=400, detail="Database constraint violation while updating zone.")
        
    return zone

@router.delete("/{id}", status_code=200, summary="Delete Zone")
async def delete_zone(id: int, session: Session = Depends(get_session)):
    zone = session.get(Zone, id)
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found.")
        
    from app.db.models.camera import Camera
    in_use_by_camera = session.exec(
        select(Camera).where((Camera.zone_id == id) | (Camera.zone == zone.name))
    ).first()
    
    if in_use_by_camera:
        raise HTTPException(status_code=409, detail="Zone is in use by one or more cameras.")
        
    session.delete(zone)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail="Database constraint violation while deleting zone.")
        
    return {"message": "row deleted successfully"}

@router.patch("/{id}/toggle", response_model=ZoneResponse, summary="Toggle Zone")
async def toggle_zone(id: int, session: Session = Depends(get_session)):
    zone = session.get(Zone, id)
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found.")
        
    zone.enabled = not zone.enabled
    session.add(zone)
    try:
        session.commit()
        session.refresh(zone)
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=400, detail="Database constraint violation while toggling zone.")
        
    return zone
