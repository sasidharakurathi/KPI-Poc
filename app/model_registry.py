
import logging
import threading

import numpy as np
from ultralytics import YOLO

from .config import settings

logger = logging.getLogger(__name__)

_models: dict[str, YOLO] = {}
_lock = threading.Lock()


def _warm_up(model: YOLO) -> None:
    """Run one dummy inference to move the model onto the target device (GPU
    VRAM when CUDA is available). Without this, YOLO(path) leaves weights in
    CPU RAM; the move to VRAM only happens on the first real .predict() call."""
    device = settings.DEVICE
    half = settings.USE_HALF and device != "cpu"
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=64, device=device, half=half, verbose=False)


def get_model(model_path: str) -> YOLO:
    """Returns the shared model for this path, loading it on first use if
    it wasn't already preloaded. KPI detectors should always call this
    instead of constructing YOLO(path) directly."""
    with _lock:
        model = _models.get(model_path)
        if model is None:
            logger.info(f"[model_registry] loading {model_path}")
            model = YOLO(model_path)
            _warm_up(model)
            _models[model_path] = model
        return model


def preload_all(model_paths: set[str]) -> None:
    """Eagerly loads every given path into the registry. Call once from the
    FastAPI lifespan startup hook so the first video job doesn't pay for it."""
    for path in sorted(model_paths):
        get_model(path)
    logger.info(f"[model_registry] {len(_models)} distinct model(s) resident")


def reset_tracker(model: YOLO) -> None:
    """Clears ByteTrack state (track IDs, Kalman filter, frame counter) on a
    shared model instance so it doesn't carry state over from a previous
    video/job, or from a different KPI's use of the same shared model file.
    Call this once, right before your own .track(persist=True, ...) loop
    starts. No-op if the model has never been used for tracking yet."""
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return
    for tracker in getattr(predictor, "trackers", []):
        tracker.reset()
