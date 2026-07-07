"""Worker 2 — runs the KPI pipeline on recorded clips as they arrive.

Clips are handed off one at a time by stream_recorder's per-camera capture
threads. Processed strictly sequentially (one clip at a time, regardless of
how many cameras are recording in parallel) since every KPI shares the same
GPU — this keeps VRAM usage bounded rather than growing with camera count.

Mirrors the exact job-creation and KPI-resolution logic the manual
/api/videos/upload endpoint uses for a camera_id, so a recorded clip is
processed with precisely the KPIs configured for that camera. Each clip is
deleted after processing (success or failure) — its alerts already have
their own saved frames in storage/alerts, so the raw clip has no further use.

Also feeds the adaptive_controller: the total wall-clock time to run every
KPI on a clip (combined, not per-KPI) is measured here and reported against
that clip's actual duration, closing the feedback loop that lets each
camera's KPIs automatically thin out their own frame sampling under load
(see app/adaptive_controller.py and BaseKPI._should_process_frame).

Reliability note: one slow or failing clip must never take down the ones
behind it. run_pipeline() already swallows every exception per-KPI and never
raises to its caller, so a KPI bug can't stall the queue. But _process() also
does DB/setup work (job creation, camera lookup) *before* reaching
run_pipeline's own try/except — a transient failure there (DB hiccup, disk
error) is still wrapped by _run()'s own try/except below, so it's logged and
the worker moves on to the next queued clip instead of dying silently and
leaving every clip behind it stuck in the queue forever.
"""
import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Queued-but-not-yet-processed clips at/above this count triggers a warning
# log — a visible signal that processing is falling behind the live
# recording rate, well before it becomes a real problem.
_BACKLOG_WARNING_THRESHOLD = 3


@dataclass
class _ClipJob:
    camera_id: str
    camera_name: str
    clip_path: str
    duration_sec: float


class ClipProcessor:
    def __init__(self) -> None:
        self._queue: "queue.Queue[_ClipJob]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="clip-processor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)

    def enqueue(self, camera_id: str, camera_name: str, clip_path: str, duration_sec: float) -> None:
        self._queue.put(_ClipJob(camera_id, camera_name, clip_path, duration_sec))
        depth = self._queue.qsize()
        logger.info(f"[clip-processor] queued {clip_path} (camera {camera_id}, queue depth {depth})")
        if depth >= _BACKLOG_WARNING_THRESHOLD:
            logger.warning(
                f"[clip-processor] {depth} clip(s) now waiting to be processed — "
                f"falling behind the live recording rate. The adaptive controller "
                f"will widen frame skipping on affected cameras, but a persistently "
                f"growing backlog means alerts are arriving later than real time."
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._process(job)
            except Exception:
                # A clip's setup/DB work can fail for reasons unrelated to the KPI
                # pipeline itself (run_pipeline never raises — see module docstring).
                # Whatever the cause, one bad clip must never stop every clip queued
                # behind it from ever being processed.
                logger.exception(
                    f"[clip-processor] unexpected error processing {job.clip_path} — "
                    f"discarding this clip and continuing with the next one"
                )
                try:
                    os.remove(job.clip_path)
                except OSError:
                    pass
            finally:
                self._queue.task_done()

    def _process(self, job: _ClipJob) -> None:
        from . import db
        from .adaptive_controller import adaptive_controller
        from .config_loader import resolve_kpi_names
        from .job_manager import job_manager
        from .kpis import list_registered_names
        from .pipeline import run_pipeline

        cam = db.get_camera(job.camera_id)
        kpi_names_to_run = resolve_kpi_names(cam.kpi_ids) if cam else []
        implemented = set(list_registered_names())
        kpis_running = [n for n in kpi_names_to_run if n in implemented]

        job_id = str(uuid.uuid4())
        job_manager.create_job(
            job_id=job_id,
            filename=Path(job.clip_path).name,
            video_path=job.clip_path,
            camera_id=job.camera_id,
            camera_name=job.camera_name,
            kpis_running=kpis_running,
        )
        logger.info(
            f"[clip-processor] processing {job.clip_path} "
            f"(camera {job.camera_id}, kpis={kpis_running or 'none'})"
        )
        t0 = time.perf_counter()
        try:
            run_pipeline(job_id, job.clip_path, kpi_names_to_run)
        except Exception:
            logger.exception(f"[clip-processor] pipeline failed for {job.clip_path}")
        finally:
            wall_time = time.perf_counter() - t0
            adaptive_controller.record_clip_result(job.camera_id, job.duration_sec, wall_time)
            try:
                os.remove(job.clip_path)
            except OSError as exc:
                logger.warning(f"[clip-processor] failed to delete {job.clip_path}: {exc}")


clip_processor = ClipProcessor()
