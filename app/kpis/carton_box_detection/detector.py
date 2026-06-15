import cv2
import os
import supervision as sv
from ultralytics import YOLO
from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ...config import settings

@register_kpi
class BoxCounterKPI(BaseKPI):
    name = "object_detection"
    display_name = "Object Detection and Counting"
    color = (0, 255, 0)  # Green for detections

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize Supervision Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        model_path = self._get("model_path", "best.pt")
        conf_threshold = self._get("confidence", 0.75)
        model = YOLO(model_path)
        
        cap = cv2.VideoCapture(video_path)
        frame_annotations = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=conf_threshold, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # Prepare data for BaseKPI structure
            frame_dets = []
            status_lines = []
            
            if detections is not None and len(detections) > 0:
                for i in range(len(detections)):
                    box = detections.xyxy[i]
                    class_id = detections.class_id[i]
                    confidence = detections.confidence[i]
                    label_text = f"{model.names[class_id]}"
                    
                    frame_dets.append(Detection(
                        int(box[0]), int(box[1]), int(box[2]), int(box[3]), 
                        label_text, float(confidence)
                    ))
                
                status_lines.append(f"Objects Detected: {len(detections)}")

            frame_annotations.append(FrameAnnotation(frame_idx, frame_dets, status_lines))
            frame_idx += 1

        cap.release()
        return KPIResult(self.name, self.display_name, self.color, frame_annotations, {"total_frames": frame_idx})