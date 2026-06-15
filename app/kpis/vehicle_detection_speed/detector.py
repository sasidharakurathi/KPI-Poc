import cv2
import numpy as np
import supervision as sv
from collections import defaultdict, deque
from ultralytics import YOLO
from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ...config import settings

@register_kpi
class SpeedTrackerKPI(BaseKPI):
    name = "speed_tracker"
    display_name = "Vehicle Speed Tracker"
    color = (0, 255, 255)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Perspective Calibration
        self.src = np.float32([[400, 300], [900, 300], [1200, 800], [100, 800]])
        self.dst = np.float32([[0, 0], [20, 0], [20, 40], [0, 40]])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        # Using ByteTrack as per your requirement
        self.tracker = sv.ByteTrack()

    def _transform_points(self, points: np.ndarray):
        # Added safety check for empty input
        if points is None or points.size == 0:
            return None
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.M)
        return transformed.reshape(-1, 2)

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        model_path = self._get("model_path", "yolov8n.pt")
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        frame_annotations = []
        coordinates = defaultdict(lambda: deque(maxlen=int(fps)))
        alerted_vehicles = set()
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)
            
            frame_dets = []
            status_lines = []

            # Robust check: Ensure detections exist and contain tracker IDs
            if detections is not None and len(detections) > 0 and detections.tracker_id is not None:
                anchors = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                transformed_anchors = self._transform_points(anchors)

                if transformed_anchors is not None:
                    for i, track_id in enumerate(detections.tracker_id):
                        coordinates[track_id].append(transformed_anchors[i])
                        
                        speed_kmh = 0
                        if len(coordinates[track_id]) > 5:
                            dist = np.linalg.norm(coordinates[track_id][-1] - coordinates[track_id][0])
                            time_s = len(coordinates[track_id]) / fps
                            speed_kmh = int((dist / time_s) * 3.6)
                            speed_kmh = speed_kmh if speed_kmh > 5 else 0

                        # Alerting
                        if speed_kmh > 25.0 and track_id not in alerted_vehicles:
                            self._save_alert(frame, "speeding_violation", job_id, frame_idx, 
                                             confidence=float(detections.confidence[i]),
                                             extra={"speed": speed_kmh, "track_id": int(track_id)})
                            alerted_vehicles.add(track_id)

                        label = f"{speed_kmh}km/h" if speed_kmh > 0 else "warming..."
                        xyxy = detections.xyxy[i]
                        frame_dets.append(Detection(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), label, float(detections.confidence[i])))
                        status_lines.append(f"ID {track_id}: {label}")

            frame_annotations.append(FrameAnnotation(frame_idx, frame_dets, status_lines))
            frame_idx += 1

        cap.release()
        return KPIResult(self.name, self.display_name, self.color, frame_annotations, {"total_frames": frame_idx})