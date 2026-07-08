
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

# Queue depth at/above this logs a warning that processing is falling behind the live recording rate.
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
                f"falling behind the live recording rate."
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
                # One bad clip must never stop every clip queued behind it from being processed.
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
            utilization = wall_time / job.duration_sec if job.duration_sec > 0 else 0.0
            logger.info(
                f"[clip-processor] {Path(job.clip_path).name} (camera {job.camera_id}) "
                f"— all {len(kpis_running)} KPI(s) done in {wall_time:.2f}s "
                f"(clip duration {job.duration_sec:.1f}s, {utilization:.2f}x real-time)"
            )
            try:
                os.remove(job.clip_path)
            except OSError as exc:
                logger.warning(f"[clip-processor] failed to delete {job.clip_path}: {exc}")


clip_processor = ClipProcessor()
