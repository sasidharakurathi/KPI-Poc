import os
import uuid
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from app import db
from app.config import settings
from app.config_loader import resolve_kpi_names
from app.job_manager import job_manager
from app.kpis import list_registered_names
from app.pipeline import run_pipeline
from app.schemas import JobStatus, JobStatusResponse, UploadResponse
from app.video_thinning import thin_video
from app.db import query_alerts

router = APIRouter(prefix="/api/videos", tags=["videos"])

_ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _thin_then_run_pipeline(
    job_id: str, src_path: str, thinned_path: str, kpi_names: Optional[list[str]]
) -> None:
    import logging
    logger = logging.getLogger(__name__)
    try:
        thin_video(src_path, thinned_path)
    except Exception as exc:
        logger.error(f"[upload] job {job_id}: frame-thinning failed: {exc}", exc_info=True)
        job_manager.update(job_id, JobStatus.FAILED, error=f"frame-thinning failed: {exc}")
        return
    try:
        run_pipeline(job_id, thinned_path, kpi_names)
    finally:
        try:
            os.remove(thinned_path)
        except OSError as exc:
            logger.warning(f"[upload] failed to delete thinned temp file {thinned_path}: {exc}")


@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="Video file to analyse")],
    camera_id: Annotated[Optional[str], Form(description="Camera position ID")] = None,
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
        kpis_requested = resolve_kpi_names(cam.kpi_ids) or list(cam.kpi_model_ids)
        kpi_names_to_run = kpis_requested or None

    implemented = set(list_registered_names())
    kpis_running = [n for n in (kpis_requested or list_registered_names()) if n in implemented]

    job_id = str(uuid.uuid4())
    upload_path = settings.UPLOAD_DIR / f"{job_id}{suffix}"

    with upload_path.open("wb") as fp:
        while chunk := await file.read(1024 * 1024):
            fp.write(chunk)

    job = job_manager.create_job(
        job_id=job_id, filename=file.filename,
        video_path=str(upload_path), camera_id=camera_id,
        camera_name=camera_name, kpis_running=kpis_running,
    )
    thinned_path = upload_path.with_name(f"{upload_path.stem}_thinned{upload_path.suffix}")
    background_tasks.add_task(
        _thin_then_run_pipeline, job_id, str(upload_path), str(thinned_path), kpi_names_to_run
    )

    return UploadResponse(
        job_id=job_id, status=JobStatus.PENDING, filename=file.filename,
        created_at=job.created_at, camera_id=camera_id, camera_name=camera_name,
        kpis_requested=kpis_requested, kpis_running=kpis_running,
        message=(
            f"Video received for camera {camera_id} ({camera_name}). "
            f"Running {len(kpis_running)} KPI(s): {', '.join(kpis_running) or 'none'}."
            if camera_id
            else "Video received. Running all registered KPIs."
        ),
    )


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatusResponse(
        job_id=job.job_id, status=job.status, filename=job.filename,
        created_at=job.created_at, completed_at=job.completed_at,
        camera_id=job.camera_id, camera_name=job.camera_name,
        kpis_running=job.kpis_running, kpi_results=job.kpi_results,
        error=job.error,
    )


@router.get("/{job_id}/alerts")
async def get_job_alerts(job_id: str, kpi_name: Optional[str] = None, limit: int = 200):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    rows, _total = query_alerts(job_id=job_id, kpi_name=kpi_name, limit=limit)
    return rows
