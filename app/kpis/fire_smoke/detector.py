import cv2
import numpy as np

from ... import model_registry
from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ...config import settings

_BATCH_SIZE = 8


def _roi_motion_score(cur_gray: np.ndarray, prev_gray: np.ndarray, box) -> float:
    """Mean pixel change within box between two grayscale frames."""
    h, w = cur_gray.shape
    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(cv2.absdiff(cur_gray[y1:y2, x1:x2], prev_gray[y1:y2, x1:x2]).mean())


@register_kpi
class FireSmokeKPI(BaseKPI):
    name = "fire_smoke"
    display_name = "Fire & Smoke"

    def setup(self, video_path: str, job_id: str = "") -> None:
        self._job_id = job_id
        self.device = settings.DEVICE
        self.half   = settings.USE_HALF and self.device != "cpu"

        self.model_path       = self._get("model_path",           settings.FIRE_SMOKE_MODEL_PATH or "app/models/fire-smoke.pt")
        self.smoke_conf       = self._get("smoke_confidence",     0.28)
        self.fire_conf        = self._get("fire_confidence",      0.50)
        self.alarm_threshold  = self._get("alarm_frame_threshold",15)
        self.fire_threshold   = self._get("fire_frame_threshold",  5)
        alarm_hold_secs       = self._get("alarm_hold_seconds",   3.0)
        self.frame_stride     = max(1, self._get("frame_stride",  2))
        self.min_smoke_motion = self._get("min_smoke_motion",    2.5)
        self.min_fire_motion  = self._get("min_fire_motion",     2.5)
        self.infer_imgsz      = self._get("infer_imgsz",         640)

        self.model = model_registry.get_model(self.model_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        self.hold_frames = int(alarm_hold_secs * fps)

        self.smoke_persistence  = 0
        self.smoke_alarm_active = False
        self.last_smoke_frame   = -1
        self.fire_persistence   = 0
        self.fire_alarm_active  = False
        self.last_fire_frame    = -1
        self.prev_gray: "np.ndarray | None" = None

        self.smoke_events = 0
        self.fire_events  = 0
        self._frames_seen = 0
        self.batch: list[tuple[int, np.ndarray]] = []

    def _process_one(self, fidx: int, frame: np.ndarray, results) -> None:
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        smoke_this = fire_this = False
        smoke_boxes = []
        fire_boxes  = []

        for box in results.boxes:
            cls_id   = int(box.cls[0])
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls_id == 0 and conf_val >= self.smoke_conf:
                if self.prev_gray is not None and _roi_motion_score(
                    cur_gray, self.prev_gray, (x1, y1, x2, y2)
                ) < self.min_smoke_motion:
                    continue
                smoke_this = True
                smoke_boxes.append((x1, y1, x2, y2, f"smoke {conf_val:.2f}", (128, 128, 128)))
            elif cls_id == 1 and conf_val >= self.fire_conf:
                if self.prev_gray is not None and _roi_motion_score(
                    cur_gray, self.prev_gray, (x1, y1, x2, y2)
                ) < self.min_fire_motion:
                    continue
                fire_this = True
                fire_boxes.append((x1, y1, x2, y2, f"fire {conf_val:.2f}", (0, 80, 255)))

        self.prev_gray = cur_gray

        if smoke_this:
            self.smoke_persistence += 1
            self.last_smoke_frame   = fidx
            if self.smoke_persistence >= self.alarm_threshold and not self.smoke_alarm_active:
                self.smoke_alarm_active = True
                self.smoke_events += 1
                best_conf = max((float(b[4].split()[-1]) for b in smoke_boxes), default=self.smoke_conf)
                self._save_alert(
                    "smoke_alarm", self._job_id, fidx,
                    confidence=best_conf,
                    extra={"persistence": self.smoke_persistence},
                    boxes=smoke_boxes,
                )
        else:
            self.smoke_persistence = max(0, self.smoke_persistence - 1)
            if self.smoke_alarm_active and (fidx - self.last_smoke_frame) > self.hold_frames:
                self.smoke_alarm_active = False
                self.smoke_persistence  = 0

        if fire_this:
            self.fire_persistence += 1
            self.last_fire_frame   = fidx
            if self.fire_persistence >= self.fire_threshold and not self.fire_alarm_active:
                self.fire_alarm_active = True
                self.fire_events += 1
                best_conf = max((float(b[4].split()[-1]) for b in fire_boxes), default=self.fire_conf)
                self._save_alert(
                    "fire_detected", self._job_id, fidx,
                    confidence=best_conf,
                    extra={"persistence": self.fire_persistence},
                    boxes=fire_boxes,
                )
        else:
            self.fire_persistence = max(0, self.fire_persistence - 1)
            if self.fire_alarm_active and (fidx - self.last_fire_frame) > self.hold_frames:
                self.fire_alarm_active = False
                self.fire_persistence  = 0

    def _flush_batch(self) -> None:
        if not self.batch:
            return
        frames = [f for _, f in self.batch]
        results_list = self.model.predict(
            frames, conf=self.smoke_conf, imgsz=self.infer_imgsz,
            device=self.device, half=self.half, verbose=False,
        )
        for (fidx, frame), results in zip(self.batch, results_list):
            self._process_one(fidx, frame, results)
        self.batch = []

    def process_frame(self, frame_idx: int, frame: np.ndarray, job_id: str = "") -> None:
        self._observe(frame, frame_idx, self._job_id)

        if frame_idx % self.frame_stride == 0:
            self.batch.append((frame_idx, frame))
            if len(self.batch) >= _BATCH_SIZE:
                self._flush_batch()

        self._frames_seen = frame_idx + 1

    def finalize(self) -> KPIResult:
        self._flush_batch()
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "smoke_alarm_events": self.smoke_events,
            "fire_alarm_events":  self.fire_events,
            "alarm_triggered":    self.smoke_events > 0 or self.fire_events > 0,
            "total_frames":       self._frames_seen,
            "device":             self.device,
        })
