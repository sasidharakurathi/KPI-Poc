"""Job manager backed by the SQLModel DB layer."""
from datetime import datetime
from typing import Any, Optional

from .db import Job, get_job, update_job, upsert_job
from .schemas import JobStatus


class _JobProxy:
    """Thin wrapper so callers can use job.status, job.kpi_results etc."""

    def __init__(self, row: Job) -> None:
        self._row = row

    def __getattr__(self, name: str) -> Any:
        return getattr(self._row, name)

    @property
    def status(self) -> JobStatus:
        return JobStatus(self._row.status)


class JobManager:
    """Persists jobs in PostgreSQL via SQLModel. State survives restarts."""

    def create_job(
        self,
        job_id: str,
        filename: str,
        video_path: str,
        camera_id: Optional[str] = None,
        camera_name: Optional[str] = None,
        kpis_running: Optional[list[str]] = None,
    ) -> _JobProxy:
        row = upsert_job(
            job_id=job_id,
            filename=filename,
            video_path=video_path,
            camera_id=camera_id,
            camera_name=camera_name,
            kpis_running=kpis_running or [],
        )
        return _JobProxy(row)

    def get(self, job_id: str) -> Optional[_JobProxy]:
        row = get_job(job_id)
        return _JobProxy(row) if row else None

    def update(
        self,
        job_id: str,
        status: JobStatus,
        kpi_results: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        update_job(
            job_id=job_id,
            status=status.value,
            kpi_results=kpi_results,
            error=error,
        )


job_manager = JobManager()
