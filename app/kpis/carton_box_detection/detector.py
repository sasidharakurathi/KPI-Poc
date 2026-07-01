import cv2
from ultralytics import YOLO

from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ...config import settings


@register_kpi
class BoxCounterKPI(BaseKPI):
    name = "object_detection"
    display_name = "Object Detection and Counting"

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path   = self._get("model_path",  "app/models/carton-box-detection.pt")
        conf         = self._get("confidence",  0.75)
        frame_stride = max(1, self._get("frame_stride", 2))

        model = YOLO(model_path)
        cap   = cv2.VideoCapture(video_path)

        alert_fired = False
        alert_events = 0
        frame_idx    = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self._observe(frame, frame_idx, job_id)

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            results = model(frame, conf=conf, device=device, half=half, verbose=False)
            r = results[0]

            if len(r.boxes) > 0 and not alert_fired:
                alert_fired = True
                alert_events += 1
                n = len(r.boxes)
                boxes = []
                for j in range(n):
                    bx = r.boxes.xyxy[j].int().tolist()
                    cls_id = int(r.boxes.cls[j])
                    label  = str(model.names.get(cls_id, "obj"))
                    boxes.append((*bx, label, (0, 255, 0)))
                self._save_alert(
                    "objects_detected", job_id, frame_idx,
                    extra={"object_count": n},
                    boxes=boxes,
                )

            frame_idx += 1

        cap.release()
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "alert_events": alert_events,
            "total_frames": frame_idx,
            "device":       device,
        })
