import cv2
import numpy as np
import supervision as sv
from collections import defaultdict, deque
from ultralytics import YOLO

from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ...config import settings


@register_kpi
class SpeedTrackerKPI(BaseKPI):
    name = "speed_tracker"
    display_name = "Vehicle Speed Tracker"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Perspective transform calibration points (pixels → metres)
        self.src = np.float32([[400, 300], [900, 300], [1200, 800], [100, 800]])
        self.dst = np.float32([[0, 0], [20, 0], [20, 40], [0, 40]])
        self.M   = cv2.getPerspectiveTransform(self.src, self.dst)
        self.tracker = sv.ByteTrack()

    def _transform_points(self, points: np.ndarray) -> np.ndarray:
        reshaped    = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.M)
        return transformed.reshape(-1, 2)

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path    = self._get("model_path", "app/models/vehicle-detection-speed.pt")
        conf          = self._get("confidence",  0.50)
        speed_limit   = self._get("speed_limit_kmh", 25.0)

        model = YOLO(model_path)
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0

        coordinates:     dict = defaultdict(lambda: deque(maxlen=int(fps)))
        alerted_vehicles: set[int] = set()
        alert_events = 0
        frame_idx    = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self._observe(frame, frame_idx, job_id)

            results  = model(frame, conf=conf, device=device, half=half, verbose=False)[0]
            sv_dets  = sv.Detections.from_ultralytics(results)
            sv_dets  = self.tracker.update_with_detections(sv_dets)

            if sv_dets.tracker_id is not None and len(sv_dets) > 0:
                anchors     = sv_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                transformed = self._transform_points(anchors)

                for i, track_id in enumerate(sv_dets.tracker_id):
                    coordinates[track_id].append(transformed[i])

                    speed_kmh = 0
                    if len(coordinates[track_id]) > 5:
                        dist      = float(np.linalg.norm(coordinates[track_id][-1] - coordinates[track_id][0]))
                        time_s    = len(coordinates[track_id]) / fps
                        speed_kmh = int((dist / time_s) * 3.6)
                        speed_kmh = speed_kmh if speed_kmh > 5 else 0

                    if speed_kmh > speed_limit and track_id not in alerted_vehicles:
                        alerted_vehicles.add(track_id)
                        alert_events += 1
                        xyxy = sv_dets.xyxy[i]
                        x1, y1, x2, y2 = map(int, xyxy)
                        self._save_alert(
                            "speeding_violation", job_id, frame_idx,
                            confidence=float(sv_dets.confidence[i]) if sv_dets.confidence is not None else 1.0,
                            extra={"speed_kmh": speed_kmh, "track_id": int(track_id)},
                            boxes=[(x1, y1, x2, y2, f"{speed_kmh}km/h", (0, 255, 255))],
                        )

            frame_idx += 1

        cap.release()
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "alert_events":    alert_events,
            "vehicles_tracked": len(coordinates),
            "total_frames":    frame_idx,
            "device":          device,
        })
