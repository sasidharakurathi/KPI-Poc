import threading
from ultralytics import YOLO

_RAW_CONF = 0.05


class SharedInference:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, YOLO] = {}
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
            model = self._models.get(model_path)
            if model is None:
                model = YOLO(model_path)
                self._models[model_path] = model
            result = model.predict(
                frame, conf=_RAW_CONF, imgsz=imgsz, device=device, half=half, verbose=False
            )[0]
            boxes = [
                (*box.xyxy[0].tolist(), int(box.cls[0]), float(box.conf[0]))
                for box in result.boxes
            ]
            self._cache[key] = boxes
            return boxes
