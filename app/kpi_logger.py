"""
KPI compute logger — per-model resource usage log.

Writes logs/kpi_run_{job_id}_{timestamp}.json after each completed job.

One entry per .pt model file.  Metrics are scoped to the current process
(not system-wide) so other applications do not pollute the numbers.

  cpu.avg_percent / peak_percent   — process CPU %, sampled every 500 ms
  ram.before_mb / after_mb / delta_mb / avg_percent / peak_percent
                                   — process RSS memory
  gpu.util_avg_percent / peak_percent — GPU compute % via torch.cuda.utilization
  gpu.mem_before_mb / peak_mb / after_mb / delta_mb / avg_percent / peak_percent
                                   — GPU allocated memory

NOTE: KPIs run concurrently in a ThreadPoolExecutor, so sibling KPIs running
at the same time will slightly raise the GPU/RAM baseline for each other.
CPU is process-level, so it reflects only this process but across all threads.
Multi-model KPIs (PPE, Floating, MobileUsage) show combined metrics per model
because inference for both models happens inside the same process_video() call.
"""

import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import psutil
import torch

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"

# ---------------------------------------------------------------------------
# nvidia-smi helper (cached; works on Windows, Linux, Mac with NVIDIA GPU)
# ---------------------------------------------------------------------------

_NVIDIASMI_CMD: Optional[list] = None   # None = unchecked; [] = not available


def _get_nvidiasmi() -> Optional[list]:
    """Return the nvidia-smi command list, or None if unavailable. Cached."""
    global _NVIDIASMI_CMD
    if _NVIDIASMI_CMD is not None:
        return _NVIDIASMI_CMD or None

    candidates = ["nvidia-smi"]
    if sys.platform == "win32":
        candidates += [
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ]

    for cmd in candidates:
        try:
            r = subprocess.run([cmd, "--version"], capture_output=True, timeout=3)
            if r.returncode == 0:
                _NVIDIASMI_CMD = [cmd]
                return _NVIDIASMI_CMD
        except Exception:
            pass

    _NVIDIASMI_CMD = []
    return None


def _smi_gpu_util(gpu_idx: int) -> Optional[float]:
    """Query GPU compute utilisation % via nvidia-smi. Returns None if unavailable."""
    cmd = _get_nvidiasmi()
    if not cmd:
        return None
    try:
        r = subprocess.run(
            cmd + [f"--id={gpu_idx}",
                   "--query-gpu=utilization.gpu",
                   "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip())
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class SystemInfo:
    os: str
    python_version: str
    cpu_model: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    total_ram_gb: float
    device: str
    gpu_name: Optional[str]
    gpu_total_memory_gb: Optional[float]
    cuda_version: Optional[str]
    torch_version: str
    ultralytics_version: str


@dataclass
class CpuStats:
    avg_percent: float
    peak_percent: float


@dataclass
class RamStats:
    before_mb: float
    after_mb: float
    delta_mb: float
    avg_percent: float      # % of total system RAM (process RSS / total RAM)
    peak_percent: float


@dataclass
class GpuStats:
    util_avg_percent: Optional[float]
    util_peak_percent: Optional[float]
    mem_before_mb: Optional[float]
    mem_peak_mb: Optional[float]
    mem_after_mb: Optional[float]
    mem_delta_mb: Optional[float]
    mem_avg_percent: Optional[float]   # % of total VRAM
    mem_peak_percent: Optional[float]


@dataclass
class ModelRunLog:
    model_file: str             # e.g. "ppl-count-yolo26m.pt"
    model_path: str             # full path as in config
    model_size_mb: Optional[float]
    used_by_kpi: str
    inference_time_sec: float
    frames_processed: int
    fps_throughput: float
    cpu: CpuStats
    ram: RamStats
    gpu: GpuStats
    status: str                 # "success" | "failed"
    error: Optional[str]


@dataclass
class PipelineRunLog:
    job_id: str
    timestamp_utc: str
    video_path: str
    total_pipeline_sec: float
    thread_workers: int
    system: SystemInfo
    models: list[ModelRunLog]   # one entry per .pt file
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Background resource sampler (process-level)
# ---------------------------------------------------------------------------

class _ResourceSampler:
    """
    Polls process-level CPU %, process RSS %, GPU utilisation %, and GPU
    allocated memory every `interval` seconds on a daemon thread.

    Each instance owns its own psutil.Process() object so concurrent samplers
    (one per KPI) never race each other's cpu_percent() internal timer.

    GPU backend detection order:
      CUDA (NVIDIA) → torch.cuda.utilization + memory_allocated
      MPS  (Apple)  → torch.mps.current_allocated_memory (util not available)
    Both are collected in separate try/except blocks so a util failure does
    not silently drop the memory sample.
    """

    def __init__(self, device: str, interval: float = 0.25) -> None:
        self._device   = device
        self._interval = interval
        self._total_ram_mb: float = psutil.virtual_memory().total / (1024 ** 2)
        # Logical core count — used to normalise cpu_percent to 0-100 %
        self._cpu_count: int = psutil.cpu_count(logical=True) or 1

        # Per-instance Process object — avoids shared-state race between parallel KPIs
        self._proc = psutil.Process(os.getpid())

        # ── CUDA (NVIDIA) ───────────────────────────────────────────────────
        self._cuda = (
            torch.cuda.is_available()
            and device.startswith("cuda")
        )
        self._gpu_idx      = 0
        self._gpu_total_mb = 0.0
        if self._cuda:
            try:
                self._gpu_idx = int(device.split(":")[-1]) if ":" in device else 0
                props = torch.cuda.get_device_properties(self._gpu_idx)
                self._gpu_total_mb = props.total_memory / (1024 ** 2)
            except Exception:
                self._cuda = False

        # ── Apple MPS ───────────────────────────────────────────────────────
        self._mps = (
            not self._cuda
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and device == "mps"
        )

        # nvidia-smi sampled every ~2 s to avoid subprocess overhead
        self._smi_every: int = max(1, int(2.0 / interval))
        self._smi_tick:  int = 0

        self._cpu_samples:     list[float] = []
        self._ram_pct_samples: list[float] = []
        self._gpu_util:        list[float] = []   # empty on MPS (not exposed)
        self._gpu_mem_mb:      list[float] = []

        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── public ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._cpu_samples     = []
        self._ram_pct_samples = []
        self._gpu_util        = []
        self._gpu_mem_mb      = []
        self._running         = True
        # Warm up this instance's cpu_percent counter (first call always returns 0)
        self._proc.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

        def _agg(s: list) -> tuple[float, float]:
            if not s:
                return 0.0, 0.0
            return round(sum(s) / len(s), 1), round(max(s), 1)

        cpu_avg, cpu_peak = _agg(self._cpu_samples)
        ram_avg, ram_peak = _agg(self._ram_pct_samples)

        out: dict = {
            "cpu_avg":        cpu_avg,
            "cpu_peak":       cpu_peak,
            "ram_avg_pct":    ram_avg,
            "ram_peak_pct":   ram_peak,
            "gpu_util_avg":   None,
            "gpu_util_peak":  None,
            "gpu_mem_peak_mb": None,
            "gpu_mem_avg_pct": None,
            "gpu_mem_peak_pct": None,
        }

        if self._gpu_util:
            gu_avg, gu_peak = _agg(self._gpu_util)
            out["gpu_util_avg"]  = gu_avg
            out["gpu_util_peak"] = gu_peak

        if self._gpu_mem_mb:
            gm_peak = round(max(self._gpu_mem_mb), 1)
            gm_avg  = round(sum(self._gpu_mem_mb) / len(self._gpu_mem_mb), 1)
            out["gpu_mem_peak_mb"] = gm_peak
            if self._gpu_total_mb > 0:
                out["gpu_mem_avg_pct"]  = round(gm_avg  / self._gpu_total_mb * 100, 1)
                out["gpu_mem_peak_pct"] = round(gm_peak / self._gpu_total_mb * 100, 1)

        return out

    # ── internal ─────────────────────────────────────────────────────────────

    def _poll(self) -> None:
        # CUDA: pin this thread to the right device so memory_allocated() is reliable
        if self._cuda:
            try:
                torch.cuda.set_device(self._gpu_idx)
            except Exception:
                pass

        while self._running:
            # Sleep FIRST so each sample covers a full interval window.
            # (Sampling immediately after start() gives a ~0 ms delta → cpu_percent = 0.)
            time.sleep(self._interval)
            if not self._running:
                break

            # ── CPU ─────────────────────────────────────────────────────────
            # Normalise by logical core count so value is always 0–100 %.
            # (psutil reports per-core %, so a fully saturated 16-core machine
            #  returns 1600 % without normalisation.)
            try:
                raw = self._proc.cpu_percent(interval=None)
                self._cpu_samples.append(round(raw / self._cpu_count, 1))
            except Exception:
                pass

            # ── RAM ─────────────────────────────────────────────────────────
            try:
                rss_mb = self._proc.memory_info().rss / (1024 ** 2)
                self._ram_pct_samples.append(round(rss_mb / self._total_ram_mb * 100, 2))
            except Exception:
                pass

            # ── GPU utilisation % ────────────────────────────────────────────
            # Kept in a separate block from memory so a failure here never
            # prevents the memory sample from being collected.
            if self._cuda:
                util_got = False

                # 1) torch.cuda.utilization — fast, but returns 0 on some WDDM
                #    laptop GPUs even when the GPU is actively running kernels.
                try:
                    val = float(torch.cuda.utilization(self._gpu_idx))
                    if val > 0:
                        self._gpu_util.append(val)
                        util_got = True
                except Exception:
                    pass

                # 2) pynvml direct (same underlying NVML, but avoids torch wrapper bugs)
                if not util_got:
                    try:
                        import pynvml               # type: ignore[import]
                        pynvml.nvmlInit()
                        h = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_idx)
                        val = float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                        if val > 0:
                            self._gpu_util.append(val)
                            util_got = True
                    except Exception:
                        pass

                # 3) nvidia-smi subprocess — most reliable on WDDM (laptop GPUs),
                #    but heavy: only call every ~2 s.
                if not util_got:
                    self._smi_tick += 1
                    if self._smi_tick >= self._smi_every:
                        self._smi_tick = 0
                        val = _smi_gpu_util(self._gpu_idx)
                        if val is not None:
                            self._gpu_util.append(val)

            # ── GPU memory ──────────────────────────────────────────────────
            if self._cuda:
                try:
                    self._gpu_mem_mb.append(
                        torch.cuda.memory_allocated(self._gpu_idx) / (1024 ** 2)
                    )
                except Exception:
                    pass
            elif self._mps:
                try:
                    self._gpu_mem_mb.append(
                        torch.mps.current_allocated_memory() / (1024 ** 2)
                    )
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_size_mb(path: str) -> Optional[float]:
    try:
        return round(os.path.getsize(path) / (1024 ** 2), 2)
    except (OSError, TypeError):
        return None


def _proc_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def _gpu_allocated_mb(device: str) -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        idx = int(device.split(":")[-1]) if ":" in device else 0
        return round(torch.cuda.memory_allocated(idx) / (1024 ** 2), 1)
    except Exception:
        return None


def _total_ram_mb() -> float:
    return psutil.virtual_memory().total / (1024 ** 2)


def _extract_model_paths(kpi_cfg: dict) -> list[str]:
    """All model file paths from a KPI config section (any key containing 'model_path')."""
    return sorted(
        v for k, v in kpi_cfg.items()
        if "model_path" in k and isinstance(v, str) and v
    )


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def collect_system_info(device: str) -> SystemInfo:
    gpu_name = gpu_total_gb = cuda_ver = None

    if torch.cuda.is_available():
        idx = int(device.split(":")[-1]) if ":" in device else 0
        try:
            gpu_name     = torch.cuda.get_device_name(idx)
            props        = torch.cuda.get_device_properties(idx)
            gpu_total_gb = round(props.total_memory / (1024 ** 3), 2)
            cuda_ver     = torch.version.cuda
        except Exception:
            pass

    try:
        import ultralytics
        ul_ver = ultralytics.__version__
    except Exception:
        ul_ver = "unknown"

    return SystemInfo(
        os=f"{platform.system()} {platform.release()}",
        python_version=sys.version.split()[0],
        cpu_model=platform.processor() or platform.machine(),
        cpu_physical_cores=psutil.cpu_count(logical=False) or 0,
        cpu_logical_cores=psutil.cpu_count(logical=True) or 0,
        total_ram_gb=round(psutil.virtual_memory().total / (1024 ** 3), 2),
        device=device,
        gpu_name=gpu_name,
        gpu_total_memory_gb=gpu_total_gb,
        cuda_version=cuda_ver,
        torch_version=torch.__version__,
        ultralytics_version=ul_ver,
    )


# ---------------------------------------------------------------------------
# Per-KPI metrics collector (context manager used in pipeline._run_kpi)
# ---------------------------------------------------------------------------

class KPIMetricsCollector:
    """
    Wraps one KPI's process_video() call and measures resource usage.
    After __exit__, call .to_model_logs() to get one ModelRunLog per .pt file.
    """

    def __init__(self, kpi, device: str) -> None:
        self._kpi     = kpi
        self._device  = device
        self._sampler = _ResourceSampler(device)
        self._ram0:   float = 0.0
        self._gpu0:   Optional[float] = None
        self._stats:  dict = {}
        self._ram1:   float = 0.0
        self._gpu1:   Optional[float] = None
        # Filled by record()
        self._elapsed: float = 0.0
        self._frames:  int   = 0
        self._fps:     float = 0.0
        self._status:  str   = "failed"
        self._error:   Optional[str] = None
        self._summary: dict  = {}

    def __enter__(self) -> "KPIMetricsCollector":
        self._ram0 = _proc_rss_mb()
        self._gpu0 = _gpu_allocated_mb(self._device)
        self._sampler.start()
        return self

    def __exit__(self, *_) -> None:
        self._stats = self._sampler.stop()
        self._ram1  = _proc_rss_mb()
        self._gpu1  = _gpu_allocated_mb(self._device)

    def record(
        self,
        elapsed: float,
        result,                         # KPIResult | None
        error: Optional[str] = None,
    ) -> None:
        self._elapsed = elapsed
        self._frames  = result.summary.get("total_frames", 0) if result else 0
        self._fps     = round(self._frames / elapsed, 2) if elapsed > 0 else 0.0
        self._status  = "success" if error is None else "failed"
        self._error   = error
        self._summary = result.summary if result else {}

    def to_model_logs(self) -> list[ModelRunLog]:
        """Expand this KPI's metrics into one ModelRunLog entry per .pt file."""
        s      = self._stats
        ram_mb = _total_ram_mb()
        gpu1   = self._gpu1
        gpu0   = self._gpu0

        cpu = CpuStats(
            avg_percent  = s["cpu_avg"],
            peak_percent = s["cpu_peak"],
        )
        ram = RamStats(
            before_mb    = round(self._ram0, 1),
            after_mb     = round(self._ram1, 1),
            delta_mb     = round(self._ram1 - self._ram0, 1),
            avg_percent  = s["ram_avg_pct"],
            peak_percent = s["ram_peak_pct"],
        )
        gpu = GpuStats(
            util_avg_percent  = s["gpu_util_avg"],
            util_peak_percent = s["gpu_util_peak"],
            mem_before_mb     = gpu0,
            mem_peak_mb       = s["gpu_mem_peak_mb"],
            mem_after_mb      = gpu1,
            mem_delta_mb      = round(gpu1 - gpu0, 1) if gpu1 is not None and gpu0 is not None else None,
            mem_avg_percent   = s["gpu_mem_avg_pct"],
            mem_peak_percent  = s["gpu_mem_peak_pct"],
        )

        model_paths = _extract_model_paths(self._kpi._cfg)
        if not model_paths:
            # Fallback: no model_path in config — still emit one entry for the KPI
            model_paths = [f"<unknown — {self._kpi.name}>"]

        return [
            ModelRunLog(
                model_file        = Path(p).name,
                model_path        = p,
                model_size_mb     = _file_size_mb(p),
                used_by_kpi       = self._kpi.name,
                inference_time_sec= round(self._elapsed, 3),
                frames_processed  = self._frames,
                fps_throughput    = self._fps,
                cpu               = cpu,
                ram               = ram,
                gpu               = gpu,
                status            = self._status,
                error             = self._error,
            )
            for p in model_paths
        ]


# ---------------------------------------------------------------------------
# Log writers
# ---------------------------------------------------------------------------

def write_pipeline_log(log: PipelineRunLog) -> Path:
    """Write the full detailed log (JSON) and the compact summary (CSV + JSON)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = LOGS_DIR / f"kpi_run_{log.job_id}_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(log), f, indent=2, default=str)

    logger.info(f"[kpi_logger] detailed log → {out_path}")

    # Also write the compact summary alongside
    try:
        _write_summary(log, ts)
    except Exception as e:
        logger.warning(f"[kpi_logger] summary log failed: {e}")

    return out_path


def _write_summary(log: PipelineRunLog, ts: str) -> None:
    """
    Write a compact per-model summary as both CSV and JSON.

    Columns:
      model_file | kpi | size_mb | time_sec | fps |
      cpu_peak_% | ram_delta_mb | gpu_util_peak_% | gpu_mem_peak_mb
    GPU columns are included only when the system has a GPU.
    """
    import csv

    has_gpu = log.system.gpu_name is not None

    rows: list[dict] = []
    for m in log.models:
        row: dict = {
            "model_file":      m.model_file,
            "kpi":             m.used_by_kpi,
            "size_mb":         m.model_size_mb,
            "time_sec":        m.inference_time_sec,
            "fps":             m.fps_throughput,
            "cpu_peak_%":      m.cpu.peak_percent,
            "ram_delta_mb":    m.ram.delta_mb,
        }
        if has_gpu:
            row["gpu_util_peak_%"] = m.gpu.util_peak_percent   # null if unavailable
            row["gpu_mem_peak_mb"] = m.gpu.mem_peak_mb         # null if unavailable
        rows.append(row)

    stem = f"kpi_summary_{log.job_id}_{ts}"

    # ── CSV ─────────────────────────────────────────────────────────────────
    csv_path = LOGS_DIR / f"{stem}.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── JSON ─────────────────────────────────────────────────────────────────
    json_path = LOGS_DIR / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)

    logger.info(f"[kpi_logger] summary  → {csv_path.name} + {json_path.name}")
