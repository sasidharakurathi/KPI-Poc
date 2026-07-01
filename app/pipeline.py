import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from .job_manager import job_manager
from .kpis import get_registered_kpis
from .kpis.base import KPIResult
from .schemas import JobStatus
from .config import settings
from .kpi_logger import (
    KPIMetricsCollector,
    ModelRunLog,
    PipelineRunLog,
    collect_system_info,
    write_pipeline_log,
)

logger = logging.getLogger(__name__)


def _run_kpi(
    kpi, video_path: str, job_id: str, device: str
) -> tuple[str, Optional[KPIResult], float, KPIMetricsCollector, Optional[Exception]]:
    logger.info(f"[{kpi.name}] starting")
    t0 = time.perf_counter()
    exc_store: Optional[Exception] = None
    result: Optional[KPIResult] = None

    with KPIMetricsCollector(kpi, device) as coll:
        try:
            result = kpi.process_video(video_path, job_id=job_id)
        except Exception as exc:
            exc_store = exc
        finally:
            elapsed = time.perf_counter() - t0
            coll.record(elapsed, result, error=str(exc_store) if exc_store else None)

    if exc_store:
        logger.error(f"[{kpi.name}] failed in {elapsed:.2f}s: {exc_store}", exc_info=exc_store)
    else:
        logger.info(f"[{kpi.name}] done in {elapsed:.2f}s — {result.summary}")

    return kpi.name, result, elapsed, coll, exc_store


def run_pipeline(
    job_id: str,
    video_path: str,
    kpi_names: Optional[list[str]] = None,
) -> None:
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
        model_logs: list[ModelRunLog] = []
        workers = min(len(kpis), settings.MAX_WORKERS)
        device = settings.DEVICE

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_kpi, kpi, video_path, job_id, device): kpi.name
                for kpi in kpis
            }
            for future in as_completed(futures):
                kpi_name = futures[future]
                try:
                    name, result, elapsed, coll, exc = future.result()
                    if coll:
                        model_logs.extend(coll.to_model_logs())
                    if exc:
                        logger.error(f"[{kpi_name}] KPI failed — excluded from output")
                    else:
                        kpi_results[name] = result
                        kpi_timings[name] = round(elapsed, 2)
                except Exception as exc:
                    logger.error(f"[{kpi_name}] unexpected executor error: {exc}", exc_info=True)

        if not kpi_results:
            raise RuntimeError("All KPIs failed — no results produced.")

        total_elapsed = time.perf_counter() - pipeline_start
        logger.info(
            f"[pipeline] job {job_id} completed in {total_elapsed:.2f}s — "
            f"KPIs: {kpi_timings}"
        )

        try:
            run_log = PipelineRunLog(
                job_id=job_id,
                timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                video_path=video_path,
                total_pipeline_sec=round(total_elapsed, 3),
                compose_time_sec=0.0,
                thread_workers=workers,
                system=collect_system_info(device),
                models=model_logs,
                notes=[
                    "CPU/RAM are process-level (this PID only). KPIs run concurrently so "
                    "sibling KPIs slightly raise each other's baseline.",
                    "Detections save an 8-frame raw clip window to disk + DB; no video output.",
                ],
            )
            write_pipeline_log(run_log)
        except Exception as log_exc:
            logger.warning(f"[kpi_logger] failed to write run log: {log_exc}")

        summaries = {name: r.summary for name, r in kpi_results.items()}
        job_manager.update(job_id, JobStatus.COMPLETED, kpi_results=summaries)

    except Exception as exc:
        logger.error(f"[pipeline] job {job_id} failed: {exc}", exc_info=True)
        job_manager.update(job_id, JobStatus.FAILED, error=str(exc))
