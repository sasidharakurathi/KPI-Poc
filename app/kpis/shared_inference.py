import threading

from .. import model_registry

_RAW_CONF = 0.05


class SharedInference:
    """Per-frame raw-detection cache on top of model_registry's preloaded,
    process-wide model instances — this class only avoids recomputing the
    same (model, frame, imgsz) forward pass twice within a single job."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[tuple, list] = {}

    def predict_boxes(
        self, model_path: str, frame_idx: int, frame, imgsz: int, device: str, half: bool
    ) -> list[tuple[float, float, float, float, int, float]]:
        """Returns (x1, y1, x2, y2, cls_id, conf) tuples for this model+frame+imgsz."""
        key = (model_path, frame_idx, imgsz)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            model = model_registry.get_model(model_path)
            result = model.predict(
                frame, conf=_RAW_CONF, imgsz=imgsz, device=device, half=half, verbose=False
            )[0]
            boxes = [
                (*box.xyxy[0].tolist(), int(box.cls[0]), float(box.conf[0]))
                for box in result.boxes
            ]
            self._cache[key] = boxes
            return boxes
