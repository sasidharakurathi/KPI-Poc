"""
kpi/base.py
Shared base class and thread primitives for all KPI modules.
"""
import threading
import time
import cv2
from typing import Optional
from ultralytics import YOLO


class BaseKPIConfig:
    """Shared detection/tracking parameters used by every KPI module."""
    video_source:     str   = "videos/open_crowd.mp4"
    model_path:       str   = "models/yolo26m.pt"
    conf_threshold:   float = 0.35
    min_box_area:     int   = 800
    max_pillar_ratio: float = 4.0
    min_person_ratio: float = 0.6
    delay_ms:         int   = 33


class BaseKPIModule:
    """
    Thread-safe KPI base — handles model loading, video capture,
    per-frame geometry filtering, and frame encoding.
    Subclasses override _process_frame() to add their own analytics.
    """

    TARGET_CLASS_ID = 0  # 'person' in all supported models

    def __init__(self):
        self._running = False
        self._frame: Optional[bytes] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # ── Public lifecycle ──────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        with self._lock:
            self._frame = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        with self._lock:
            self._frame = None

    def join(self, timeout: float = 5.0) -> None:
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    # ── Subclass contract ─────────────────────────────────────────────────────

    def _get_config(self) -> BaseKPIConfig:
        raise NotImplementedError

    def _reset_state(self) -> None:
        """Called on video loop-back."""
        pass

    def _process_frame(self, frame, valid_detections: list, frame_num: int, fps: float) -> None:
        """
        Called once per frame with pre-filtered detections.
        valid_detections: list of dicts — {track_id, conf, x1, y1, x2, y2}
        """
        raise NotImplementedError

    # ── Shared geometry filter ────────────────────────────────────────────────

    def _filter_boxes(self, boxes, cfg: BaseKPIConfig) -> list:
        """Apply all filters from app.py; return list of clean detection dicts."""
        valid = []
        if boxes is None or boxes.id is None:
            return valid

        track_ids   = boxes.id.int().cpu().tolist()
        cls_indices = boxes.cls.int().cpu().tolist()
        xyxy        = boxes.xyxy.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        for i in range(len(track_ids)):
            if cls_indices[i] != self.TARGET_CLASS_ID:
                continue
            if confidences[i] < cfg.conf_threshold:
                continue

            x1, y1, x2, y2 = xyxy[i]
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            area   = w * h
            aspect = h / w

            if area   < cfg.min_box_area:     continue
            if aspect > cfg.max_pillar_ratio:  continue
            if aspect < cfg.min_person_ratio:  continue

            valid.append({
                "track_id": track_ids[i],
                "conf":     confidences[i],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
        return valid

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _loop(self) -> None:
        cfg = self._get_config()

        try:
            model = YOLO(cfg.model_path)
        except Exception as exc:
            print(f"[{self.__class__.__name__}] Model load failed: {exc}")
            self._running = False
            return

        src = cfg.video_source.strip()
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if not cap.isOpened():
            print(f"[{self.__class__.__name__}] Cannot open video: {src!r}")
            self._running = False
            return

        frame_num = 0
        t0        = time.time()
        self._reset_state()

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_num = 0
                    self._reset_state()
                    continue

                frame_num += 1
                live_fps   = frame_num / (time.time() - t0 + 1e-9)

                results = model.track(
                    source=frame, persist=True,
                    tracker="bytetrack.yaml", verbose=False,
                )[0]

                valid = self._filter_boxes(results.boxes, cfg)
                self._process_frame(frame, valid, frame_num, live_fps)

                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self._frame = buf.tobytes()

                time.sleep(cfg.delay_ms / 1000.0)

        except Exception as exc:
            print(f"[{self.__class__.__name__}] Pipeline error: {exc}")
        finally:
            cap.release()
            self._running = False
            print(f"[{self.__class__.__name__}] Loop stopped.")
