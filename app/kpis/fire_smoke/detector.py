import cv2
from ultralytics import YOLO

from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ...config import settings

_DEFAULT_MODEL_PATH        = settings.FIRE_SMOKE_MODEL_PATH
_DEFAULT_SMOKE_CONF        = 0.28
_DEFAULT_FIRE_CONF         = 0.50
_DEFAULT_ALARM_THRESHOLD   = 15
_DEFAULT_ALARM_HOLD_SECS   = 3.0


@register_kpi
class FireSmokeKPI(BaseKPI):
    name = "fire_smoke"
    display_name = "Fire & Smoke"
    color = (0, 0, 255)  # red (BGR)

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path      = self._get("model_path", _DEFAULT_MODEL_PATH)
        smoke_conf      = self._get("smoke_confidence", _DEFAULT_SMOKE_CONF)
        fire_conf       = self._get("fire_confidence", _DEFAULT_FIRE_CONF)
        alarm_threshold = self._get("alarm_frame_threshold", _DEFAULT_ALARM_THRESHOLD)
        alarm_hold_secs = self._get("alarm_hold_seconds", _DEFAULT_ALARM_HOLD_SECS)

        model = YOLO(model_path)

        cap = cv2.VideoCapture(video_path)
        fps         = cap.get(cv2.CAP_PROP_FPS) or 25
        hold_frames = int(alarm_hold_secs * fps)

        frame_annotations: list[FrameAnnotation] = []

        smoke_persistence  = 0
        smoke_alarm_active = False
        last_smoke_frame   = -1

        fire_alarm_active  = False
        last_fire_frame    = -1

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                source=frame,
                device=device,
                half=half,
                conf=smoke_conf,
                verbose=False,
            )

            detections: list[Detection] = []
            smoke_this_frame = False
            fire_this_frame  = False

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if cls_id == 0 and conf >= smoke_conf:
                        smoke_this_frame = True
                        detections.append(Detection(x1, y1, x2, y2, "smoke", conf))
                    elif cls_id == 1 and conf >= fire_conf:
                        fire_this_frame = True
                        detections.append(Detection(
                            x1, y1, x2, y2, "fire", conf,
                            color=(0, 80, 255),
                        ))

            # Smoke alarm
            if smoke_this_frame:
                smoke_persistence += 1
                last_smoke_frame   = frame_idx
                if smoke_persistence >= alarm_threshold and not smoke_alarm_active:
                    smoke_alarm_active = True
                    if job_id:
                        self._save_alert(
                            frame, "smoke_alarm", job_id, frame_idx,
                            extra={"persistence": smoke_persistence},
                            detections=detections,
                        )
            else:
                smoke_persistence = max(0, smoke_persistence - 1)
                if smoke_alarm_active and (frame_idx - last_smoke_frame) > hold_frames:
                    smoke_alarm_active = False
                    smoke_persistence  = 0

            # Fire alarm
            if fire_this_frame:
                last_fire_frame = frame_idx
                if not fire_alarm_active:
                    fire_alarm_active = True
                    fire_conf_val = max(
                        (float(box.conf[0]) for r in results for box in r.boxes
                         if int(box.cls[0]) == 1 and float(box.conf[0]) >= fire_conf),
                        default=fire_conf,
                    )
                    if job_id:
                        self._save_alert(
                            frame, "fire_detected", job_id, frame_idx,
                            confidence=fire_conf_val,
                            detections=detections,
                        )
            elif fire_alarm_active and (frame_idx - last_fire_frame) > hold_frames:
                fire_alarm_active = False

            status_lines: list[str] = []
            if smoke_alarm_active:
                status_lines.append("!! SMOKE ALARM ACTIVE")
            if fire_alarm_active:
                status_lines.append("!! FIRE ALARM ACTIVE")

            frame_annotations.append(FrameAnnotation(
                frame_idx=frame_idx,
                detections=detections,
                status_lines=status_lines,
            ))
            frame_idx += 1

        cap.release()

        frames_with_smoke = sum(
            1 for fa in frame_annotations
            if any(d.label == "smoke" for d in fa.detections)
        )
        frames_with_fire = sum(
            1 for fa in frame_annotations
            if any(d.label == "fire" for d in fa.detections)
        )

        return KPIResult(
            kpi_name=self.name,
            display_name=self.display_name,
            color=self.color,
            frame_annotations=frame_annotations,
            summary={
                "frames_with_smoke":  frames_with_smoke,
                "frames_with_fire":   frames_with_fire,
                "alarm_triggered":    frames_with_smoke > 0 or frames_with_fire > 0,
                "total_frames":       frame_idx,
                "device":             device,
            },
        )
