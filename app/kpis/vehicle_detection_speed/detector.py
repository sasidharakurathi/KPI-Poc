import cv2
import numpy as np
from ultralytics import YOLO
from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ...config import settings

@register_kpi
class SpeedTrackerKPI(BaseKPI):
    name = "speed_tracker"
    display_name = "Vehicle Speed Tracker"
    color = (0, 255, 255)  # Yellow for estimating

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half = settings.USE_HALF and device != "cpu"

        model_path = self._get("model_path", "best.pt")
        conf = self._get("confidence", 0.5)
        # Perspective Calibration
        src = np.float32([[400, 400], [800, 400], [1000, 720], [200, 720]])
        dst = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
        M = cv2.getPerspectiveTransform(src, dst)
        meters_per_pixel = 20.0 / 400.0

        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        frame_annotations = []
        vehicle_data = {}
        # Track if an alert has already been sent for this vehicle to prevent spamming
        alerted_vehicles = set()
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            
            detections = []
            status_lines = []

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, ids):
                    cx, cy = int((box[0] + box[2]) / 2), int(box[3])
                    point = np.array([[[cx, cy]]], dtype="float32")
                    wx, wy = cv2.perspectiveTransform(point, M)[0][0]

                    if track_id not in vehicle_data:
                        vehicle_data[track_id] = {'start_pos': (wx, wy), 'start_frame': frame_idx, 'speed': 0, 'locked': False}
                    
                    v_data = vehicle_data[track_id]
                    pixel_dist = np.sqrt((wx - v_data['start_pos'][0])**2 + (wy - v_data['start_pos'][1])**2)
                    meters = pixel_dist * meters_per_pixel
                    time_s = (frame_idx - v_data['start_frame']) / fps
                    
                    if time_s > 0:
                        v_data['speed'] = (meters / time_s) * 3.6
                        if meters > 10.0:
                            v_data['locked'] = True
                    
                    # ALERT LOGIC: Trigger if speed > 25kmph
                    if v_data['speed'] > 25.0 and track_id not in alerted_vehicles:
                        self._save_alert(
                            frame,
                            "speeding_violation",
                            job_id,
                            frame_idx,
                            confidence=float(results[0].boxes.conf[0]),
                            extra={"speed": round(v_data['speed'], 2), "track_id": int(track_id)}
                        )
                        alerted_vehicles.add(track_id)

                    detections.append(Detection(int(box[0]), int(box[1]), int(box[2]), int(box[3]), f"{int(v_data['speed'])}km/h", 1.0))
                    status_lines.append(f"ID {track_id}: {int(v_data['speed'])} km/h")

            frame_annotations.append(FrameAnnotation(frame_idx, detections, status_lines))
            frame_idx += 1

        cap.release()
        return KPIResult(self.name, self.display_name, self.color, frame_annotations, {"total_frames": frame_idx})