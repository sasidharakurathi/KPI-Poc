from fastapi import APIRouter

from app.kpis import list_registered_names

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok", "registered_kpis": list_registered_names()}
