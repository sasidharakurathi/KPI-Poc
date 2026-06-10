import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from .compositor import compose_video
from .job_manager import job_manager
from .kpis import get_registered_kpis
from .kpis.base import KPIResult
from .schemas import JobStatus
from .config import settings

logger = logging.getLogger(__name__)


def _run_kpi(kpi, video_path: str, job_id: str) -> tuple[str, KPIResult, float]:
    logger.info(f"[{kpi.name}] starting")
    t0 = time.perf_counter()
    result = kpi.process_video(video_path, job_id=job_id)
    elapsed = time.perf_counter() - t0
    logger.info(f"[{kpi.name}] done in {elapsed:.2f}s — {result.summary}")
    return kpi.name, result, elapsed


def run_pipeline(
    job_id: str,
    video_path: str,
    output_path: str,
    kpi_names: Optional[list[str]] = None,
) -> None:
    """
    Entry point called as a FastAPI background task.

    Args:
        kpi_names: If provided, only the KPIs whose name is in this list
                   will run. Pass None to run every registered KPI (default).
    """
    job_manager.update(job_id, JobStatus.PROCESSING)
    pipeline_start = time.perf_counter()
    logger.info(f"[pipeline] job {job_id} started — filter: {kpi_names or 'all'}")

    try:
        all_kpis = get_registered_kpis()

        if kpi_names is not None:
            kpis = [k for k in all_kpis if k.name in kpi_names]
            skipped = [k.name for k in all_kpis if k.name not in kpi_names]
            if skipped:
                logger.info(f"[pipeline] skipping KPIs not in camera mapping: {skipped}")
        else:
            kpis = all_kpis

        if not kpis:
            raise RuntimeError(
                "No KPIs to run. "
                "Either none are registered, all are disabled, or none match the camera's KPI list."
            )

        kpi_results: dict[str, KPIResult] = {}
        kpi_timings: dict[str, float] = {}
        workers = min(len(kpis), settings.MAX_WORKERS)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_kpi, kpi, video_path, job_id): kpi.name
                for kpi in kpis
            }
            for future in as_completed(futures):
                kpi_name = futures[future]
                try:
                    name, result, elapsed = future.result()
                    kpi_results[name] = result
                    kpi_timings[name] = round(elapsed, 2)
                except Exception as exc:
                    logger.error(f"[{kpi_name}] failed: {exc}", exc_info=True)

        if not kpi_results:
            raise RuntimeError("All KPIs failed — no results to compose.")

        compose_start = time.perf_counter()
        logger.info(f"[pipeline] compositing {len(kpi_results)} KPI result(s)")
        compose_video(video_path, kpi_results, output_path)
        compose_elapsed = time.perf_counter() - compose_start

        total_elapsed = time.perf_counter() - pipeline_start
        logger.info(
            f"[pipeline] job {job_id} completed in {total_elapsed:.2f}s — "
            f"KPIs: {kpi_timings} | compose: {compose_elapsed:.2f}s → {output_path}"
        )

        summaries = {name: r.summary for name, r in kpi_results.items()}
        job_manager.update(
            job_id,
            JobStatus.COMPLETED,
            output_path=output_path,
            kpi_results=summaries,
        )

    except Exception as exc:
        logger.error(f"[pipeline] job {job_id} failed: {exc}", exc_info=True)
        job_manager.update(job_id, JobStatus.FAILED, error=str(exc))
