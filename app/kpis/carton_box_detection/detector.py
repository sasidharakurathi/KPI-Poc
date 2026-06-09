import cv2
from ultralytics import YOLO
from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ...config import settings

# Default values
_DEFAULT_CONF = 0.75

@register_kpi
class BoxCounterKPI(BaseKPI):
    name = "box_counter"
    display_name = "Box Counter"
    color = (0, 255, 0)  # BGR green

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half = settings.USE_HALF and device != "cpu"

        # Load parameters
        model_path = self._get("model_path", "best.pt")
        conf = self._get("confidence", _DEFAULT_CONF)

        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        
        frame_annotations = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run prediction
            results = model.predict(
                source=frame, conf=conf, device=device, half=half, verbose=False
            )
            
            detections = []
            status_lines = []

            for r in results:
                box_count = len(r.boxes)
                
                # Logic for triggering the alert
                if box_count > 0:
                    self._save_alert(
                        frame, 
                        "box_detected", 
                        job_id, 
                        frame_idx,
                        confidence=float(r.boxes.conf.max()) if len(r.boxes) > 0 else 0.0, 
                        extra={"count": box_count}
                    )

                # Extract detection data for the UI
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = r.names[int(box.cls[0])]
                    conf_val = float(box.conf[0])
                    detections.append(Detection(x1, y1, x2, y2, label, conf_val))
                
                status_lines.append(f"Count: {box_count}")

            # Store annotations
            frame_annotations.append(FrameAnnotation(
                frame_idx=frame_idx,
                detections=detections,
                status_lines=status_lines,
            ))
            frame_idx += 1

        cap.release()

        return KPIResult(
            kpi_name=self.name,
            display_name=self.display_name,
            color=self.color,
            frame_annotations=frame_annotations,
            summary={"total_frames": frame_idx, "final_count": len(results[0].boxes) if results else 0},
        )