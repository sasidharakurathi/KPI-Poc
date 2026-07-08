import threading

from .. import model_registry

_RAW_CONF = 0.05


def _boxes_from_result(result) -> list[tuple[float, float, float, float, int, float]]:
    return [
        (*box.xyxy[0].tolist(), int(box.cls[0]), float(box.conf[0]))
        for box in result.boxes
    ]


class SharedInference:
    """Per-frame raw-detection cache, so KPIs sharing a job don't recompute the same (model, frame, imgsz) forward pass twice."""

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
            boxes = _boxes_from_result(result)
            self._cache[key] = boxes
            return boxes

    def predict_boxes_batch(
        self, model_path: str, frame_batch: list[tuple[int, "object"]], imgsz: int, device: str, half: bool
    ) -> dict[int, list[tuple[float, float, float, float, int, float]]]:
        to_run: list[tuple[int, object]] = []
        result_map: dict[int, list] = {}
        with self._lock:
            for frame_idx, frame in frame_batch:
                key = (model_path, frame_idx, imgsz)
                cached = self._cache.get(key)
                if cached is not None:
                    result_map[frame_idx] = cached
                else:
                    to_run.append((frame_idx, frame))

        if to_run:
            model = model_registry.get_model(model_path)
            frames = [f for _, f in to_run]
            results = model.predict(
                frames, conf=_RAW_CONF, imgsz=imgsz, device=device, half=half, verbose=False
            )
            with self._lock:
                for (frame_idx, _), result in zip(to_run, results):
                    boxes = _boxes_from_result(result)
                    self._cache[(model_path, frame_idx, imgsz)] = boxes
                    result_map[frame_idx] = boxes

        return result_map
