import cv2
import numpy as np
from ultralytics import YOLO

from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ...config import settings


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

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path       = self._get("model_path",           settings.FIRE_SMOKE_MODEL_PATH or "app/models/fire-smoke.pt")
        smoke_conf       = self._get("smoke_confidence",     0.28)
        fire_conf        = self._get("fire_confidence",      0.50)
        alarm_threshold  = self._get("alarm_frame_threshold",15)
        fire_threshold   = self._get("fire_frame_threshold",  5)
        alarm_hold_secs  = self._get("alarm_hold_seconds",   3.0)
        frame_stride     = max(1, self._get("frame_stride",  2))
        min_smoke_motion = self._get("min_smoke_motion",    2.5)
        min_fire_motion  = self._get("min_fire_motion",     2.5)
        infer_imgsz      = self._get("infer_imgsz",         640)

        model = YOLO(model_path)
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        hold_frames = int(alarm_hold_secs * fps)

        smoke_persistence  = 0
        smoke_alarm_active = False
        last_smoke_frame   = -1
        fire_persistence   = 0
        fire_alarm_active  = False
        last_fire_frame    = -1
        prev_gray: "np.ndarray | None" = None

        smoke_events = fire_events = 0
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self._observe(frame, frame_idx, job_id)

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            results  = model.predict(frame, conf=smoke_conf, imgsz=infer_imgsz,
                                    device=device, half=half, verbose=False)
            cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            smoke_this = fire_this = False
            smoke_boxes = []
            fire_boxes  = []

            for r in results:
                for box in r.boxes:
                    cls_id   = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if cls_id == 0 and conf_val >= smoke_conf:
                        if prev_gray is not None and _roi_motion_score(
                            cur_gray, prev_gray, (x1, y1, x2, y2)
                        ) < min_smoke_motion:
                            continue
                        smoke_this = True
                        smoke_boxes.append((x1, y1, x2, y2, f"smoke {conf_val:.2f}", (128, 128, 128)))
                    elif cls_id == 1 and conf_val >= fire_conf:
                        if prev_gray is not None and _roi_motion_score(
                            cur_gray, prev_gray, (x1, y1, x2, y2)
                        ) < min_fire_motion:
                            continue
                        fire_this = True
                        fire_boxes.append((x1, y1, x2, y2, f"fire {conf_val:.2f}", (0, 80, 255)))

            prev_gray = cur_gray

            # smoke logic
            if smoke_this:
                smoke_persistence += 1
                last_smoke_frame   = frame_idx
                if smoke_persistence >= alarm_threshold and not smoke_alarm_active:
                    smoke_alarm_active = True
                    smoke_events += 1
                    best_conf = max((float(b[4].split()[-1]) for b in smoke_boxes), default=smoke_conf)
                    self._save_alert(
                        "smoke_alarm", job_id, frame_idx,
                        confidence=best_conf,
                        extra={"persistence": smoke_persistence},
                        boxes=smoke_boxes,
                    )
            else:
                smoke_persistence = max(0, smoke_persistence - 1)
                if smoke_alarm_active and (frame_idx - last_smoke_frame) > hold_frames:
                    smoke_alarm_active = False
                    smoke_persistence  = 0

            if fire_this:
                fire_persistence += 1
                last_fire_frame   = frame_idx
                if fire_persistence >= fire_threshold and not fire_alarm_active:
                    fire_alarm_active = True
                    fire_events += 1
                    best_conf = max((float(b[4].split()[-1]) for b in fire_boxes), default=fire_conf)
                    self._save_alert(
                        "fire_detected", job_id, frame_idx,
                        confidence=best_conf,
                        extra={"persistence": fire_persistence},
                        boxes=fire_boxes,
                    )
            else:
                fire_persistence = max(0, fire_persistence - 1)
                if fire_alarm_active and (frame_idx - last_fire_frame) > hold_frames:
                    fire_alarm_active = False
                    fire_persistence  = 0

            frame_idx += 1

        cap.release()
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "smoke_alarm_events": smoke_events,
            "fire_alarm_events":  fire_events,
            "alarm_triggered":    smoke_events > 0 or fire_events > 0,
            "total_frames":       frame_idx,
            "device":             device,
        })
