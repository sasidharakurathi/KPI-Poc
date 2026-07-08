
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .config import settings

logger = logging.getLogger(__name__)

_models: dict[str, YOLO] = {}
_lock = threading.Lock()

_WARMUP_SPECS: dict[str, tuple[int, bool]] = {
    "ppe":               (512, False),
    "yolo26s-pose":      (512, False),
    "fire-smoke":        (512, False),
    "yolo26m-pose":      (640, True),   # dynamic shape; .track() is how both its consumers call it
    "yolo26m":           (512, False),
    "cigarette":         (320, False),
    "ppl-count-yolo26m": (640, True),
    "anpr_lpr":           (640, False),
    "carton-box-detection": (640, False),
}

# Matches each detector's _BATCH_SIZE, so the TensorRT batch profile is bound at startup, not mid-job.
_WARMUP_BATCH = 8

# Matches split-job KPI concurrency, so each worker thread's first CUDA call is paid at startup too.
_WARMUP_THREADS = 6


def _warm_up(model: YOLO, model_path: str) -> None:
    device = settings.DEVICE
    half = settings.USE_HALF and device != "cpu"
    stem = Path(model_path).stem
    imgsz, use_track = _WARMUP_SPECS.get(stem, (64, False))
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    dummy_batch = [dummy] * _WARMUP_BATCH
    if use_track:
        model.track(dummy_batch, persist=True, tracker="bytetrack.yaml", imgsz=imgsz, device=device, half=half, verbose=False)
        reset_tracker(model)   # this warmup call must not leave fake track state behind
    else:
        model.predict(dummy_batch, imgsz=imgsz, device=device, half=half, verbose=False)


def _resolve_weights_path(model_path: str) -> str:
    engine_path = Path(model_path).with_suffix(".engine")
    return str(engine_path) if engine_path.exists() else model_path


def get_model(model_path: str) -> YOLO:
    with _lock:
        model = _models.get(model_path)
        if model is None:
            resolved = _resolve_weights_path(model_path)
            logger.info(f"[model_registry] loading {resolved}")
            model = YOLO(resolved)
            _warm_up(model, model_path)
            _models[model_path] = model
        return model


def preload_all(model_paths: set[str]) -> None:
    for path in sorted(model_paths):
        get_model(path)

    # Re-warm across several threads so each one's first CUDA call is paid now, not on the first real job.
    paths = sorted(model_paths)
    with ThreadPoolExecutor(max_workers=_WARMUP_THREADS, thread_name_prefix="warmup") as ex:
        futures = [ex.submit(_warm_up, _models[p], p) for p in paths]
        for fut in futures:
            fut.result()

    logger.info(f"[model_registry] {len(_models)} distinct model(s) resident")


def reset_tracker(model: YOLO) -> None:
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return
    for tracker in getattr(predictor, "trackers", []):
        tracker.reset()
