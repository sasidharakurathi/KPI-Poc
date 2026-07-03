"""
Preloaded YOLO model registry.

Loads every distinct model file used by the *enabled* KPIs once, at server
startup (see main.py's lifespan hook), instead of each KPI re-loading its
own model(s) from disk on every single video job. This is the standard
pattern for serving ML models behind an API — pay the load cost once at
boot, not once per request (FastAPI's own docs recommend exactly this via
the lifespan context manager).

Several KPIs already point at the exact same file (yolo26m.pt,
yolo26m-pose.pt, cigarette.pt) — preloading means those are loaded once
total, not once per KPI that uses them.

Concurrency note: safe under this app's MAX_WORKERS=1 (KPIs run
sequentially, never call a shared model concurrently). `.predict()` calls
are stateless and safe to share regardless. `.track(persist=True)` calls
are NOT stateless — see reset_tracker() below.
"""
import logging
import threading

from ultralytics import YOLO

logger = logging.getLogger(__name__)

_models: dict[str, YOLO] = {}
_lock = threading.Lock()


def get_model(model_path: str) -> YOLO:
    """Returns the shared model for this path, loading it on first use if
    it wasn't already preloaded. KPI detectors should always call this
    instead of constructing YOLO(path) directly."""
    with _lock:
        model = _models.get(model_path)
        if model is None:
            logger.info(f"[model_registry] loading {model_path}")
            model = YOLO(model_path)
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
