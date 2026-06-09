import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ...config import settings

# ── Default values — used when the key is absent from config.json ─────────────
_DEFAULT_CONF              = 0.35
_DEFAULT_IOU               = 0.50
_DEFAULT_ZONE_SQFT         = 500.0   # real-world area of the monitored zone
_DEFAULT_MAX_OCCUPANCY     = 50      # occupancy cap before alert
_DEFAULT_ALERT_HOLD_FRAMES = 4       # hysteresis: consecutive frames before alert fires
_DEFAULT_THRESH = {                  # density thresholds in people-per-sq-ft
    "low":    0.05,
    "medium": 0.10,
    "high":   0.20,
}

COLOR_IN_ZONE  = (0, 0, 255)    # BGR red  — person inside the monitored zone
COLOR_OUT_ZONE = (0, 255, 0)    # BGR green — person outside the zone


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
    name         = "density_occupancy"      # unique snake_case identifier
    display_name = "Density & Occupancy"    # shown in the video overlay panel
    color        = COLOR_IN_ZONE            # BGR — red boxes for zone-occupants

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        # ── Read every parameter from config.json (second arg is the fallback) ──
        model_path        = self._get("model_path",         "best.pt")
        conf              = self._get("confidence",          _DEFAULT_CONF)
        iou               = self._get("iou_threshold",       _DEFAULT_IOU)
        zone_sqft         = self._get("zone_sqft",           _DEFAULT_ZONE_SQFT)
        max_occupancy     = self._get("max_occupancy",       _DEFAULT_MAX_OCCUPANCY)
        alert_hold_frames = self._get("alert_hold_frames",   _DEFAULT_ALERT_HOLD_FRAMES)
        thresh            = self._get("density_thresholds",  _DEFAULT_THRESH)

        # zone_points: list of [x, y] pixel coords defining the ROI polygon.
        # Falls back to None → full-frame zone is used instead.
        zone_points_raw   = self._get("zone_points", None)

        model = YOLO(model_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── Build zone polygon ─────────────────────────────────────────────────
        if zone_points_raw and len(zone_points_raw) >= 3:
            zone_pixel_pts = [tuple(p) for p in zone_points_raw]
        else:
            zone_pixel_pts = [(0, 0), (W, 0), (W, H), (0, H)]   # full frame

        zone_poly = Polygon(zone_pixel_pts)

        # ── Tracking / alert state — initialised before loop so summary{}
        #    never hits a NameError on a zero-frame video ──────────────────────
        ids_inside_zone   = set()    # IDs currently inside the zone this frame
        ids_ever_seen     = set()    # cumulative unique IDs (foot traffic)
        dwell_frames:  dict[int, int]   = {}
        consecutive_alert = 0
        alert_active      = False
        total_footfall    = 0
        density           = 0.0
        occupancy_count   = 0
        d_level           = "LOW"

        frame_annotations: list[FrameAnnotation] = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ── Run tracker ────────────────────────────────────────────────────
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf,
                iou=iou,
                classes=[0],       # person class only
                device=device,
                half=half,
                verbose=False,
            )

            # Guard: model.track() can return None or an empty list
            if not results:
                frame_annotations.append(FrameAnnotation(
                    frame_idx=frame_idx,
                    detections=[],
                    status_lines=[
                        f"Zone count:   0",
                        f"Foot traffic: {total_footfall}",
                    ],
                    extra={
                        "density_level":   d_level,
                        "alert_active":    alert_active,
                        "zone_pixel_pts":  zone_pixel_pts,
                        "people_in_zone":  0,
                        "occupancy_count": occupancy_count,
                        "total_footfall":  total_footfall,
                        "density":         0.0,
                    },
                ))
                frame_idx += 1
                continue

            result = results[0]

            detections:   list[Detection] = []
            status_lines: list[str]       = []
            people_in_zone = 0
            boxes          = result.boxes

            # Guard: boxes or track IDs may be absent on frames with no detections
            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                xyxy_list = boxes.xyxy.int().cpu().tolist()
                confs     = boxes.conf.cpu().tolist()

                for i, tid in enumerate(track_ids):
                    x1, y1, x2, y2 = xyxy_list[i]
                    conf_val = confs[i]

                    # Foot-point: bottom-centre of the bounding box
                    foot_x = int((x1 + x2) / 2)
                    foot_y = int(y2)

                    # ── Zone membership via foot-point ─────────────────────────
                    in_zone = zone_poly.contains(Point(foot_x, foot_y))

                    if in_zone:
                        people_in_zone           += 1
                        dwell_frames[tid]         = dwell_frames.get(tid, 0) + 1
                        ids_inside_zone.add(tid)
                    else:
                        ids_inside_zone.discard(tid)

                    ids_ever_seen.add(tid)

                    dwell_s   = dwell_frames.get(tid, 0) / fps
                    box_color = COLOR_IN_ZONE if in_zone else COLOR_OUT_ZONE
                    label     = (
                        f"ID:{tid} {conf_val:.2f} "
                        f"dwell:{dwell_s:.1f}s"
                    )
                    # Pass box_color so the renderer draws in-zone vs out-zone
                    # boxes in different colours (COLOR_IN_ZONE / COLOR_OUT_ZONE)
                    detections.append(
                        Detection(x1, y1, x2, y2, label, conf_val, color=box_color)
                    )

            # ── Metrics (computed every frame, outside the boxes guard) ────────
            density         = people_in_zone / max(zone_sqft, 1e-6)
            occupancy_count = len(ids_inside_zone)
            total_footfall  = len(ids_ever_seen)
            d_level         = _density_level(density, thresh)

            # ── Hysteresis-based alert (mirrors mobile ref LOW→HIGH pattern) ───
            breach = (density >= thresh["high"]) or (occupancy_count > max_occupancy)
            if breach:
                consecutive_alert += 1
            else:
                consecutive_alert = max(0, consecutive_alert - 1)

            prev_alert   = alert_active
            alert_active = consecutive_alert >= alert_hold_frames

            # Fire snapshot on the first frame of a new alert (LOW → HIGH only)
            if alert_active and not prev_alert and job_id:
                self._save_alert(
                    frame,
                    "density_occupancy_breach",
                    job_id,
                    frame_idx,
                    extra={
                        "density_level":   d_level,
                        "density":         round(density, 6),
                        "occupancy_count": occupancy_count,
                        "total_footfall":  total_footfall,
                        "people_in_zone":  people_in_zone,
                    },
                    detections=list(detections),   # snapshot of current boxes
                )

            # ── HUD status lines ───────────────────────────────────────────────
            status_lines.append(f"Zone count:  {people_in_zone}")
            status_lines.append(f"Density:     {density:.4f} ppsf")
            status_lines.append(f"Occupancy:   {occupancy_count}")
            status_lines.append(f"Foot traffic:{total_footfall}")
            status_lines.append(f"Level:       {d_level}")
            if alert_active:
                status_lines.append("⚠ ALERT: DENSITY / OCCUPANCY BREACH")

            frame_annotations.append(FrameAnnotation(
                frame_idx=frame_idx,
                detections=detections,
                status_lines=status_lines,
                extra={
                    "density_level":   d_level,
                    "alert_active":    alert_active,
                    "zone_pixel_pts":  zone_pixel_pts,
                    "people_in_zone":  people_in_zone,
                    "occupancy_count": occupancy_count,
                    "total_footfall":  total_footfall,
                    "density":         round(density, 6),
                },
            ))
            frame_idx += 1

        cap.release()

        return KPIResult(
            kpi_name=self.name,
            display_name=self.display_name,
            color=self.color,
            frame_annotations=frame_annotations,
            summary={
                "total_frames":       frame_idx,
                "total_foot_traffic": total_footfall,   # safe — initialised before loop
                "zone_sqft":          zone_sqft,
                "max_occupancy_cap":  max_occupancy,
                "device":             device,
            },
        )