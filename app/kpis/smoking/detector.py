import cv2
from ultralytics import YOLO

from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ...config import settings

_DEFAULT_MODEL_PATH       = "app/models/floating.pt"
_DEFAULT_CONF             = 0.40
_DEFAULT_ALARM_THRESHOLD  = 10        
_DEFAULT_ALARM_HOLD_SECS  = 3.0      


@register_kpi
class SmokingKPI(BaseKPI):
    name = "smoking"
    display_name = "Smoking"
    color = (0, 255, 255)  

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path      = self._get("model_path",          _DEFAULT_MODEL_PATH)
        conf            = self._get("confidence",          _DEFAULT_CONF)
        alarm_threshold = self._get("alarm_frame_threshold", _DEFAULT_ALARM_THRESHOLD)
        alarm_hold_secs = self._get("alert_hold_seconds",  _DEFAULT_ALARM_HOLD_SECS)

        model = YOLO(model_path)

        cap = cv2.VideoCapture(video_path)
        fps         = cap.get(cv2.CAP_PROP_FPS) or 25
        hold_frames = int(alarm_hold_secs * fps)

        frame_annotations: list[FrameAnnotation] = []

        smoke_persistence  = 0        
        smoke_alarm_active = False
        last_smoke_frame   = -1

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                source=frame,
                device=device,
                half=half,
                conf=conf,
                verbose=False,
            )

            detections: list[Detection] = []
            smoke_this_frame = False

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if cls_id == 0 and conf_val >= conf:
                        smoke_this_frame = True
                        detections.append(
                            Detection(x1, y1, x2, y2, "smoking", conf_val)
                        )

            if smoke_this_frame:
                smoke_persistence += 1
                last_smoke_frame = frame_idx

                if smoke_persistence >= alarm_threshold and not smoke_alarm_active:
                    smoke_alarm_active = True
                    if job_id:
                        self._save_alert(
                            frame,
                            "smoking_alarm",
                            job_id,
                            frame_idx,
                            extra={"persistence": smoke_persistence},
                        )
            else:
                smoke_persistence = max(0, smoke_persistence - 1)
                if smoke_alarm_active and (frame_idx - last_smoke_frame) > hold_frames:
                    smoke_alarm_active = False
                    smoke_persistence = 0

            status_lines: list[str] = []
            if smoke_alarm_active:
                status_lines.append("!! SMOKING ALARM ACTIVE")

            frame_annotations.append(
                FrameAnnotation(
                    frame_idx=frame_idx,
                    detections=detections,
                    status_lines=status_lines,
                )
            )
            frame_idx += 1

        cap.release()

        frames_with_smoking = sum(
            1 for fa in frame_annotations
            if any(d.label == "smoking" for d in fa.detections)
        )

        return KPIResult(
            kpi_name=self.name,
            display_name=self.display_name,
            color=self.color,
            frame_annotations=frame_annotations,
            summary={
                "frames_with_smoking": frames_with_smoking,
                "alarm_triggered":     frames_with_smoking > 0,
                "total_frames":        frame_idx,
                "device":              device,
            },
        )