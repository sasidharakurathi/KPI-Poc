import logging
import uuid
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import db
from .db import init_db, query_alerts, get_alert
from .config import settings
from .config_loader import (
    get_all as get_full_config,
    get_cameras as get_seed_cameras,
    get_kpi_config,
    get_kpi_param,
    get_kpi_registry,
    reload as reload_config,
    resolve_kpi_names,
    update_kpi_config,
)
from .job_manager import job_manager
from .kpis import get_registered_kpis, get_registry, list_registered_names
from .pipeline import run_pipeline
from .schemas import (
    CameraCreate,
    CameraInfo,
    CameraKPIDetail,
    CameraListItem,
    CameraListResponse,
    CameraUpdate,
    JobStatus,
    JobStatusResponse,
    KPIInfo,
    KPISettingsItem,
    KPISettingsResponse,
    RegisteredKPIsResponse,
    UploadResponse,
)

logging.basicConfig(level=logging.INFO)

settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.ALERTS_DIR.mkdir(parents=True, exist_ok=True)
init_db()
db.seed_cameras(get_seed_cameras())

app = FastAPI(
    title="KPI Video Analytics",
    description=(
        "Upload a video for a specific camera position, run the camera's assigned "
        "KPI models in parallel, and review the saved detection clips (8 raw frames "
        "per detection)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

_KPI_LABELS: dict[int, str] = {
    1: "Unauthorized Access",          2: "ANPR Detection",
    3: "Intrusion Detection",          4: "Vehicle Detection",
    5: "Guard Tour / Hand Detection",  6: "Industrial PPE",
    7: "Fire / Smoke Detection",       8: "Abandoned Object",
    9: "Missing Object",               10: "Intrusion",
    11: "People Count",                12: "Engagement Time / Unattended Guest",
    13: "Staff Absence",               14: "Uniform Detection",
    15: "Carton / Box Detection",      16: "Occupancy",
    17: "Usage Time",                  18: "Fallen Object",
    19: "Guard Presence",              20: "Falling Pose",
    21: "Camera Tampering",            22: "Loading / Unloading Detection",
    23: "Cleaning as per Schedule",    24: "People Density Measurement",
    25: "Opening / Closing Time",      26: "Pack Count",
    27: "Occupancy Count & Dwell Time",28: "Detect Object",
    29: "Smoking",                     30: "Floating Object / Man Overboard",
}


@app.get("/health")
async def health():
    return {"status": "ok", "registered_kpis": list_registered_names()}


@app.get("/api/kpis", response_model=RegisteredKPIsResponse)
async def list_kpis():
    kpis = get_registered_kpis()
    return RegisteredKPIsResponse(
        count=len(kpis),
        kpis=[KPIInfo(name=k.name, display_name=k.display_name) for k in kpis],
    )


@app.get("/api/config")
async def view_config():
    return get_full_config()


@app.post("/api/config/reload")
async def hot_reload_config():
    new_cfg = reload_config()
    return {"message": "config.json reloaded", "config": new_cfg}


@app.get("/api/kpis/settings", response_model=KPISettingsResponse)
async def list_kpi_settings():
    """All registered KPIs (enabled or not) with their current global config."""
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


@app.put("/api/kpis/{name}/config", response_model=KPISettingsItem)
async def update_kpi_settings(name: str, updates: Annotated[dict[str, Any], Body()]):
    """Merge partial field updates (including 'enabled') into a KPI's config block."""
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


@app.get("/api/alerts")
async def get_alerts(
    job_id: Optional[str] = None,
    kpi_name: Optional[str] = None,
    camera_id: Optional[str] = None,
    limit: int = 200,
):
    """Query saved detections. Each alert includes its saved clip frames."""
    return query_alerts(job_id=job_id, kpi_name=kpi_name, camera_id=camera_id, limit=limit)


@app.get("/api/alerts/{alert_id}")
async def get_alert_detail(alert_id: int):
    alert = get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    return alert


@app.get("/api/alerts/{alert_id}/frames/{position}")
async def get_alert_frame(alert_id: int, position: int):
    """Serve one raw frame image (0-based position) from a detection's clip window."""
    alert = get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    match = next((f for f in alert.get("frames", []) if f["position"] == position), None)
    if not match or not Path(match["path"]).exists():
        raise HTTPException(status_code=404, detail="Frame not found.")
    return FileResponse(path=match["path"], media_type="image/jpeg")


@app.get("/api/alerts/{alert_id}/labeled")
async def get_alert_labeled_frame(alert_id: int):
    """Serve the single labeled anchor-frame image (dev mode only)."""
    alert = get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    match = next((f for f in alert.get("frames", []) if f.get("labeled_path")), None)
    if not match or not Path(match["labeled_path"]).exists():
        raise HTTPException(
            status_code=404,
            detail="Labeled frame not available (job not run in developer mode).",
        )
    return FileResponse(path=match["labeled_path"], media_type="image/jpeg")


@app.get("/api/cameras", response_model=CameraListResponse)
async def list_cameras():
    registry = get_kpi_registry()
    implemented_names = set(list_registered_names())

    items = [
        CameraListItem(
            camera_id=cam.camera_id,
            name=cam.name,
            zone=cam.zone,
            priority=cam.priority,
            total_kpis=len(cam.kpi_ids),
            implemented_kpis=sum(
                1 for kid in cam.kpi_ids
                if registry.get(str(kid)) in implemented_names
            ),
        )
        for cam in db.list_cameras()
    ]
    return CameraListResponse(count=len(items), cameras=items)


@app.get("/api/cameras/{camera_id}", response_model=CameraInfo)
async def get_camera_detail(camera_id: str):
    cam = db.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return _camera_to_info(cam)


@app.post("/api/cameras", response_model=CameraInfo, status_code=201)
async def create_camera(body: CameraCreate):
    try:
        cam = db.create_camera(
            camera_id=body.camera_id, name=body.name,
            zone=body.zone, priority=body.priority, kpi_ids=body.kpi_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _camera_to_info(cam)


@app.put("/api/cameras/{camera_id}", response_model=CameraInfo)
async def update_camera(camera_id: str, body: CameraUpdate):
    cam = db.update_camera(
        camera_id, name=body.name, zone=body.zone,
        priority=body.priority, kpi_ids=body.kpi_ids,
    )
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return _camera_to_info(cam)


@app.delete("/api/cameras/{camera_id}", status_code=204)
async def delete_camera(camera_id: str):
    if not db.delete_camera(camera_id):
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")


def _camera_to_info(cam: db.Camera) -> CameraInfo:
    registry = get_kpi_registry()
    implemented_names = set(list_registered_names())
    kpis_detail = [
        CameraKPIDetail(
            kpi_id=kid,
            kpi_label=_KPI_LABELS.get(kid, f"KPI {kid}"),
            implemented=registry.get(str(kid)) in implemented_names,
        )
        for kid in cam.kpi_ids
    ]
    return CameraInfo(
        camera_id=cam.camera_id, name=cam.name, zone=cam.zone,
        priority=cam.priority, kpi_ids=cam.kpi_ids, kpis=kpis_detail,
    )


@app.post("/api/videos/upload", response_model=UploadResponse, status_code=202)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="Video file to analyse")],
    camera_id: Annotated[Optional[str], Form(description="Camera position ID (e.g. CAM-01)")] = None,
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    camera_name: Optional[str] = None
    kpis_requested: list[str] = []
    kpi_names_to_run: Optional[list[str]] = None

    if camera_id:
        cam = db.get_camera(camera_id)
        if not cam:
            raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
        camera_name = cam.name
        kpis_requested = resolve_kpi_names(cam.kpi_ids)
        kpi_names_to_run = kpis_requested

    implemented = set(list_registered_names())
    kpis_running = [n for n in (kpis_requested or list_registered_names()) if n in implemented]

    job_id = str(uuid.uuid4())
    upload_path = settings.UPLOAD_DIR / f"{job_id}{suffix}"

    with upload_path.open("wb") as fp:
        while chunk := await file.read(1024 * 1024):
            fp.write(chunk)

    job = job_manager.create_job(
        job_id=job_id,
        filename=file.filename,
        video_path=str(upload_path),
        camera_id=camera_id,
        camera_name=camera_name,
        kpis_running=kpis_running,
    )
    background_tasks.add_task(run_pipeline, job_id, str(upload_path), kpi_names_to_run)

    return UploadResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        filename=file.filename,
        created_at=job.created_at,
        camera_id=camera_id,
        camera_name=camera_name,
        kpis_requested=kpis_requested,
        kpis_running=kpis_running,
        message=(
            f"Video received for camera {camera_id} ({camera_name}). "
            f"Running {len(kpis_running)} KPI(s): {', '.join(kpis_running) or 'none'}."
            if camera_id
            else "Video received. Running all registered KPIs."
        ),
    )


@app.get("/api/videos/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        filename=job.filename,
        created_at=job.created_at,
        completed_at=job.completed_at,
        camera_id=job.camera_id,
        camera_name=job.camera_name,
        kpis_running=job.kpis_running,
        kpi_results=job.kpi_results,
        error=job.error,
    )


@app.get("/api/videos/{job_id}/alerts")
async def get_job_alerts(job_id: str, kpi_name: Optional[str] = None, limit: int = 200):
    """All detections saved for a job (each with its clip window frames)."""
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return query_alerts(job_id=job_id, kpi_name=kpi_name, limit=limit)
