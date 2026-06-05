import threading
from datetime import datetime
from typing import Any, Optional

from .schemas import JobStatus


class Job:
    def __init__(
        self,
        job_id: str,
        filename: str,
        video_path: str,
        camera_id: Optional[str] = None,
        camera_name: Optional[str] = None,
        kpis_running: Optional[list[str]] = None,
    ):
        self.job_id = job_id
        self.filename = filename
        self.video_path = video_path
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.kpis_running: list[str] = kpis_running or []
        self.status = JobStatus.PENDING
        self.created_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        self.output_path: Optional[str] = None
        self.kpi_results: Optional[dict[str, Any]] = None
        self.error: Optional[str] = None


class JobManager:
    """Thread-safe in-memory job registry. Swap for Redis/DB in production."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        job_id: str,
        filename: str,
        video_path: str,
        camera_id: Optional[str] = None,
        camera_name: Optional[str] = None,
        kpis_running: Optional[list[str]] = None,
    ) -> Job:
        job = Job(job_id, filename, video_path, camera_id, camera_name, kpis_running)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        status: JobStatus,
        output_path: Optional[str] = None,
        kpi_results: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = status
            if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                job.completed_at = datetime.utcnow()
            if output_path is not None:
                job.output_path = output_path
            if kpi_results is not None:
                job.kpi_results = kpi_results
            if error is not None:
                job.error = error


job_manager = JobManager()
