import cv2
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ..zone_labels import get_camera_zone_points
from ...config import settings

_DEFAULT_THRESH = {"low": 0.02, "medium": 0.05, "high": 0.10}


def _density_level(density: float, thresh: dict) -> str:
    if density < thresh["low"]:    return "LOW"
    if density < thresh["medium"]: return "MEDIUM"
    if density < thresh["high"]:   return "HIGH"
    return "CRITICAL"


@register_kpi
class DensityOccupancyKPI(BaseKPI):
    name         = "density_occupancy"
    display_name = "Density & Occupancy"
    requires_zone = True

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path        = self._get("model_path",         "app/models/density-occupancy.pt")
        conf              = self._get("confidence",          0.35)
        iou               = self._get("iou_threshold",       0.50)
        zone_sqft         = self._get("zone_sqft",           500.0)
        max_occupancy     = self._get("max_occupancy",       50)
        alert_hold_frames = self._get("alert_hold_frames",   4)
        thresh            = self._get("density_thresholds",  _DEFAULT_THRESH)
        zone_points_raw   = get_camera_zone_points(job_id, self.name) or self._get("zone_points", None)

        model = YOLO(model_path)
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if zone_points_raw and len(zone_points_raw) >= 3:
            zone_pts = [tuple(p) for p in zone_points_raw]
        else:
            zone_pts = [(0, 0), (W, 0), (W, H), (0, H)]
        zone_poly = Polygon(zone_pts)

        ids_inside:   set[int]        = set()
        ids_ever:     set[int]        = set()
        dwell_frames: dict[int, int]  = {}
        consecutive_alert = 0
        alert_active      = False
        total_footfall    = 0
        density           = 0.0
        occupancy_count   = 0
        alert_events      = 0
        frame_idx         = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self._observe(frame, frame_idx, job_id)

            results = model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                conf=conf, iou=iou, classes=[0],
                device=device, half=half, verbose=False,
            )

            if not results:
                frame_idx += 1
                continue

            boxes = results[0].boxes
            people_in_zone = 0

            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                xyxy_list = boxes.xyxy.int().cpu().tolist()

                for i, tid in enumerate(track_ids):
                    x1, y1, x2, y2 = xyxy_list[i]
                    foot_x = (x1 + x2) // 2
                    foot_y = y2
                    in_zone = zone_poly.contains(Point(foot_x, foot_y))

                    if in_zone:
                        people_in_zone += 1
                        dwell_frames[tid] = dwell_frames.get(tid, 0) + 1
                        ids_inside.add(tid)
                    else:
                        ids_inside.discard(tid)
                    ids_ever.add(tid)

            density         = people_in_zone / max(zone_sqft, 1e-6)
            occupancy_count = len(ids_inside)
            total_footfall  = len(ids_ever)
            d_level         = _density_level(density, thresh)

            breach = (density >= thresh["high"]) or (occupancy_count > max_occupancy)
            consecutive_alert = consecutive_alert + 1 if breach else max(0, consecutive_alert - 1)

            prev_alert   = alert_active
            alert_active = consecutive_alert >= alert_hold_frames

            if alert_active and not prev_alert:
                alert_events += 1
                self._save_alert(
                    "density_occupancy_breach", job_id, frame_idx,
                    extra={
                        "density_level":   d_level,
                        "density":         round(density, 6),
                        "occupancy_count": occupancy_count,
                        "total_footfall":  total_footfall,
                        "people_in_zone":  people_in_zone,
                    },
                )

            frame_idx += 1

        cap.release()
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "alert_events":     alert_events,
            "total_foot_traffic": total_footfall,
            "zone_sqft":        zone_sqft,
            "total_frames":     frame_idx,
            "device":           device,
        })
