import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO
from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ...config import settings

# Default values — used when the key is absent from config.json
_DEFAULT_CONF              = 0.35
_DEFAULT_IOU               = 0.50
_DEFAULT_ZONE_SQFT         = 500.0        # real-world area of the monitored zone
_DEFAULT_MAX_OCCUPANCY     = 50           # occupancy cap before alert
_DEFAULT_ALERT_HOLD_FRAMES = 4            # hysteresis: consecutive frames before alert fires
_DEFAULT_THRESH = {                       # density thresholds in people-per-sq-ft
    "low":    0.05,
    "medium": 0.10,
    "high":   0.20,
}


def _density_level(density: float, thresh: dict) -> str:
    """Map a density value to a human-readable level string."""
    if density < thresh["low"]:
        return "LOW"
    elif density < thresh["medium"]:
        return "MEDIUM"
    elif density < thresh["high"]:
        return "HIGH"
    return "CRITICAL"


@register_kpi
class DensityOccupancyKPI(BaseKPI):
    name         = "density_occupancy"          # unique snake_case identifier
    display_name = "Density & Occupancy"        # shown in the video overlay panel
    color        = (0, 0, 255)                  # BGR — red boxes for zone-occupants

    def process_video(self, video_path: str) -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        # ── Read every parameter from config.json (second arg is the fallback) ──
        model_path         = self._get("model_path",          "best.pt")
        conf               = self._get("confidence",           _DEFAULT_CONF)
        iou                = self._get("iou_threshold",        _DEFAULT_IOU)
        zone_sqft          = self._get("zone_sqft",            _DEFAULT_ZONE_SQFT)
        max_occupancy      = self._get("max_occupancy",        _DEFAULT_MAX_OCCUPANCY)
        alert_hold_frames  = self._get("alert_hold_frames",    _DEFAULT_ALERT_HOLD_FRAMES)
        thresh             = self._get("density_thresholds",   _DEFAULT_THRESH)

        # zone_points: list of [x, y] pixel coords defining the ROI polygon.
        # Falls back to None → full-frame zone is used instead.
        zone_points_raw    = self._get("zone_points", None)

        model = YOLO(model_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── Build the zone polygon ──────────────────────────────────────────
        if zone_points_raw and len(zone_points_raw) >= 3:
            zone_pixel_pts = [tuple(p) for p in zone_points_raw]
        else:
            # Default: entire frame as the monitored zone
            zone_pixel_pts = [(0, 0), (W, 0), (W, H), (0, H)]

        zone_poly = Polygon(zone_pixel_pts)

        # ── Tracking state ──────────────────────────────────────────────────
        ids_inside_zone  = set()   # IDs currently inside the zone
        ids_ever_seen    = set()   # cumulative unique IDs (foot traffic)
        dwell_frames     = {}      # tid → frame count spent in zone
        consecutive_alert = 0
        alert_active      = False
        line_in_count     = 0
        line_out_count    = 0

        frame_annotations = []
        frame_idx         = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ── Run tracker ────────────────────────────────────────────────
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf,
                iou=iou,
                classes=[0],          # person class only
                device=device,
                half=half,
                verbose=False,
            )[0]

            detections         = []
            status_lines       = []
            people_in_zone     = 0
            boxes              = results.boxes

            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                xyxy_list = boxes.xyxy.int().cpu().tolist()
                confs     = boxes.conf.cpu().tolist()

                for i, tid in enumerate(track_ids):
                    x1, y1, x2, y2 = xyxy_list[i]
                    conf_val        = confs[i]

                    # Foot-point: bottom-centre of the bounding box
                    foot_x = int((x1 + x2) / 2)
                    foot_y = int(y2)

                    # ── Zone membership via foot-point ──────────────────
                    in_zone = zone_poly.contains(Point(foot_x, foot_y))

                    if in_zone:
                        people_in_zone += 1
                        dwell_frames[tid] = dwell_frames.get(tid, 0) + 1

                    # Update persistent sets
                    if in_zone:
                        ids_inside_zone.add(tid)
                    else:
                        ids_inside_zone.discard(tid)

                    ids_ever_seen.add(tid)

                    dwell_s   = dwell_frames.get(tid, 0) / fps
                    box_color = (0, 0, 255) if in_zone else (0, 255, 0)
                    label     = (
                        f"ID:{tid} {conf_val:.2f} "
                        f"dwell:{dwell_s:.1f}s"
                    )
                    detections.append(
                        Detection(x1, y1, x2, y2, label, conf_val)
                    )

            # ── Compute density & occupancy metrics ────────────────────────
            density         = people_in_zone / max(zone_sqft, 1e-6)
            occupancy_count = len(ids_inside_zone)
            total_footfall  = len(ids_ever_seen)
            d_level         = _density_level(density, thresh)

            # ── Hysteresis-based alert logic ────────────────────────────────
            breach = (density >= thresh["high"]) or (occupancy_count > max_occupancy)
            if breach:
                consecutive_alert += 1
            else:
                consecutive_alert = max(0, consecutive_alert - 1)

            if consecutive_alert >= alert_hold_frames:
                alert_active = True
            else:
                alert_active = False

            # ── Status lines for the HUD overlay ───────────────────────────
            status_lines.append(f"Zone count:  {people_in_zone}")
            status_lines.append(f"Density:     {density:.4f} ppsf")
            status_lines.append(f"Occupancy:   {occupancy_count}")
            status_lines.append(f"Foot traffic:{total_footfall}")
            status_lines.append(f"Level:       {d_level}")
            if alert_active:
                status_lines.append("⚠ ALERT: DENSITY / OCCUPANCY BREACH")

            frame_annotations.append(
                FrameAnnotation(
                    frame_idx=frame_idx,
                    detections=detections,
                    status_lines=status_lines,
                    # Pass extra metadata so the renderer can colour-code the zone
                    extra={
                        "density_level":    d_level,
                        "alert_active":     alert_active,
                        "zone_pixel_pts":   zone_pixel_pts,
                        "people_in_zone":   people_in_zone,
                        "occupancy_count":  occupancy_count,
                        "total_footfall":   total_footfall,
                        "density":          round(density, 6),
                    },
                )
            )
            frame_idx += 1

        cap.release()

        return KPIResult(
            kpi_name=self.name,
            display_name=self.display_name,
            color=self.color,
            frame_annotations=frame_annotations,
            summary={
                "total_frames":       frame_idx,
                "total_foot_traffic": total_footfall,
                "zone_sqft":          zone_sqft,
                "max_occupancy_cap":  max_occupancy,
            },
        )