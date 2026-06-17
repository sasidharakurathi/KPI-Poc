import cv2
from ultralytics import YOLO

from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ...config import settings

_DEFAULT_MODEL_PATH     = "app/kpis/guard_presence/best.pt"
_DEFAULT_CONF           = 0.25
_DEFAULT_ALERT_HOLD     = 3.0

_CLS_GUARD = 0
_COLOR_GUARD = (0, 255, 0)
_COLOR_NON_GUARD = (0, 0, 255)


@register_kpi
class GuardPresenceKPI(BaseKPI):
    name         = "guard_presence"
    display_name = "Guard Presence"
    color        = _COLOR_GUARD

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path       = self._get("model_path",        _DEFAULT_MODEL_PATH)
        conf             = self._get("confidence",         _DEFAULT_CONF)
        alert_hold_secs  = self._get("alert_hold_seconds", _DEFAULT_ALERT_HOLD)

        model = YOLO(model_path)

        cap          = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        hold_frames  = int(alert_hold_secs * fps)

        alert_countdown     = 0
        guard_events        = 0
        in_guard_event      = False
        frame_annotations: list[FrameAnnotation] = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                source=frame,
                conf=conf,
                device=device,
                half=half,
                verbose=False,
            )

            detections: list[Detection] = []
            frame_has_guard = False

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if cls_id == 0:
                        detections.append(
                            Detection(x1, y1, x2, y2, "GUARD", conf_val, color=_COLOR_GUARD)
                        )
                        frame_has_guard = True
                    else:
                        detections.append(
                            Detection(x1, y1, x2, y2, "NON-GUARD", conf_val, color=_COLOR_NON_GUARD)
                        )

            if frame_has_guard:
                alert_countdown = hold_frames
                if not in_guard_event:
                    in_guard_event = True
                    guard_events  += 1
                    if job_id:
                        self._save_alert(
                            frame,
                            "guard_detected",
                            job_id,
                            frame_idx,
                            extra={"guard_event_number": guard_events},
                        )
            else:
                if alert_countdown > 0:
                    alert_countdown -= 1
                if alert_countdown == 0:
                    in_guard_event = False

            alert_active = alert_countdown > 0

            status_lines: list[str] = []
            if alert_active:
                status_lines.append("!! GUARD DETECTED !!")
                status_lines.append(f"Guard events: {guard_events}")

            frame_annotations.append(FrameAnnotation(
                frame_idx=frame_idx,
                detections=detections if alert_active else [],
                status_lines=status_lines,
            ))
            frame_idx += 1

        cap.release()

        return KPIResult(
            kpi_name=self.name,
            display_name=self.display_name,
            color=self.color,
            frame_annotations=frame_annotations,
            summary={
                "guard_events":  guard_events,
                "total_frames":  frame_idx,
                "device":        device,
            },
        )
