import cv2
import supervision as sv
from ultralytics import YOLO
from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ...config import settings

@register_kpi
class BoxCounterKPI(BaseKPI):
    # This name must match your database/pipeline filter string exactly
    name = "box_counter"
    display_name = "Carton Box Counter"
    color = (255, 165, 0)

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        # Load model and confidence from settings
        model_path = self._get("model_path", "best.pt")
        conf_threshold = self._get("confidence", 0.5)
        
        # Initialize YOLO and VideoCapture
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        
        frame_annotations = []
        frame_idx = 0

        # Ensure capture is opened
        if not cap.isOpened():
            return KPIResult(self.name, self.display_name, self.color, [], {"error": "Could not open video"})

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            # We use verbose=False to keep logs clean
            results = model(frame, conf=conf_threshold, verbose=False)
            
            frame_dets = []
            status_lines = []
            
            # Robust check to handle empty frames
            if results and len(results) > 0 and results[0].boxes is not None:
                # Convert results to Supervision format
                detections = sv.Detections.from_ultralytics(results[0])
                
                # If we have valid detections
                if len(detections) > 0:
                    for i in range(len(detections)):
                        xyxy = detections.xyxy[i]
                        conf = float(detections.confidence[i])
                        
                        # Add to detection list for the framework
                        frame_dets.append(Detection(
                            int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]),
                            f"Box {conf:.2f}", conf
                        ))
                    
                    status_lines.append(f"Boxes: {len(detections)}")
                else:
                    status_lines.append("No boxes detected")
            else:
                status_lines.append("No model output")

            # Always append an annotation object to maintain sync
            frame_annotations.append(FrameAnnotation(frame_idx, frame_dets, status_lines))
            frame_idx += 1

        cap.release()
        
        # Return final results
        return KPIResult(
            self.name, 
            self.display_name, 
            self.color, 
            frame_annotations, 
            {"total_frames": frame_idx}
        )