import cv2
from ultralytics import YOLO

from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ...config import settings


@register_kpi
class FireSmokeKPI(BaseKPI):
    name = "fire_smoke"
    display_name = "Fire & Smoke"

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path      = self._get("model_path",           settings.FIRE_SMOKE_MODEL_PATH or "app/models/fire-smoke.pt")
        smoke_conf      = self._get("smoke_confidence",     0.28)
        fire_conf       = self._get("fire_confidence",      0.50)
        alarm_threshold = self._get("alarm_frame_threshold",15)
        alarm_hold_secs = self._get("alarm_hold_seconds",   3.0)
        frame_stride    = max(1, self._get("frame_stride",  2))

        model = YOLO(model_path)
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        hold_frames = int(alarm_hold_secs * fps)

        smoke_persistence  = 0
        smoke_alarm_active = False
        last_smoke_frame   = -1
        fire_alarm_active  = False
        last_fire_frame    = -1

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

            results = model.predict(frame, conf=smoke_conf,
                                    device=device, half=half, verbose=False)

            smoke_this = fire_this = False
            smoke_boxes = []
            fire_boxes  = []

            for r in results:
                for box in r.boxes:
                    cls_id   = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if cls_id == 0 and conf_val >= smoke_conf:
                        smoke_this = True
                        smoke_boxes.append((x1, y1, x2, y2, f"smoke {conf_val:.2f}", (128, 128, 128)))
                    elif cls_id == 1 and conf_val >= fire_conf:
                        fire_this = True
                        fire_boxes.append((x1, y1, x2, y2, f"fire {conf_val:.2f}", (0, 80, 255)))

            # smoke logic
            if smoke_this:
                smoke_persistence += 1
                last_smoke_frame   = frame_idx
                if smoke_persistence >= alarm_threshold and not smoke_alarm_active:
                    smoke_alarm_active = True
                    smoke_events += 1
                    self._save_alert(
                        "smoke_alarm", job_id, frame_idx,
                        extra={"persistence": smoke_persistence},
                        boxes=smoke_boxes,
                    )
            else:
                smoke_persistence = max(0, smoke_persistence - 1)
                if smoke_alarm_active and (frame_idx - last_smoke_frame) > hold_frames:
                    smoke_alarm_active = False
                    smoke_persistence  = 0

            # fire logic
            if fire_this:
                last_fire_frame = frame_idx
                if not fire_alarm_active:
                    fire_alarm_active = True
                    fire_events += 1
                    best_conf = max((float(b[4].split()[-1]) for b in fire_boxes), default=fire_conf)
                    self._save_alert(
                        "fire_detected", job_id, frame_idx,
                        confidence=best_conf,
                        boxes=fire_boxes,
                    )
            elif fire_alarm_active and (frame_idx - last_fire_frame) > hold_frames:
                fire_alarm_active = False

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
