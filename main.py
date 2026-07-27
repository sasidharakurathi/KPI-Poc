"""
main.py - People Count Vision API
Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from routers.people_count     import router as people_count_router
from routers.density_occupancy import router as density_occupancy_router

app = FastAPI(
    title="People Count Vision API",
    description=(
        "Real-time people detection, tracking, density, and occupancy analytics "
        "powered by YOLOv8 + ByteTrack. Two independent KPI pipelines:\n\n"
        "• **/api/v1/people-count** - live count + cumulative footfall\n"
        "• **/api/v1/density-occupancy** - density (ppl/sqft) + occupancy alerts"
    ),
    version="2.0.0",
)

# ── CORS (restrict origins in production) ─────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(people_count_router,      prefix="/api/v1")
app.include_router(density_occupancy_router, prefix="/api/v1")


# ── Root & health ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return JSONResponse({
        "success": True,
        "message": "People Count Vision API is running",
        "version": "2.0.0",
        "docs":    "/docs",
        "endpoints": {
            # People Count KPI
            "pc_start":      "POST   /api/v1/people-count/start",
            "pc_stop":       "POST   /api/v1/people-count/stop",
            "pc_status":     "GET    /api/v1/people-count/status",
            "pc_frame":      "GET    /api/v1/people-count/frame",
            "pc_config_get": "GET    /api/v1/people-count/config",
            "pc_config_put": "PUT    /api/v1/people-count/config",
            "pc_set_video":  "PATCH  /api/v1/people-count/config/video",
            "pc_videos":     "GET    /api/v1/people-count/videos",
            "pc_models":     "GET    /api/v1/people-count/models",
            # Density / Occupancy KPI
            "do_start":      "POST   /api/v1/density-occupancy/start",
            "do_stop":       "POST   /api/v1/density-occupancy/stop",
            "do_status":     "GET    /api/v1/density-occupancy/status",
            "do_frame":      "GET    /api/v1/density-occupancy/frame",
            "do_config_get": "GET    /api/v1/density-occupancy/config",
            "do_config_put": "PUT    /api/v1/density-occupancy/config",
            "do_set_video":  "PATCH  /api/v1/density-occupancy/config/video",
            "do_set_zone":   "PATCH  /api/v1/density-occupancy/config/zone",
        }
    })


@app.get("/health")
def health():
    return JSONResponse({"success": True, "status": "healthy"})
