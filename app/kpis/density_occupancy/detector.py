import cv2
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ...config import settings

# ── Default values — used when the key is absent from config.json ─────────────
_DEFAULT_CONF              = 0.50    # raised from 0.35 — cuts weak/partial detections
_DEFAULT_IOU               = 0.50
_DEFAULT_ZONE_SQFT         = 500.0   # real-world area of the monitored zone
_DEFAULT_MAX_OCCUPANCY     = 50      # occupancy cap before alert
_DEFAULT_ALERT_HOLD_FRAMES = 4       # hysteresis: consecutive frames before alert fires
_DEFAULT_MIN_BOX_W         = 30      # px — ignore boxes narrower than this (noise/reflections)
_DEFAULT_MIN_BOX_H         = 60      # px — real standing person is always taller than this
_DEFAULT_DWELL_MIN_FRAMES  = 2       # consecutive in-zone frames required before counting
_DEFAULT_THRESH = {                  # density thresholds in people-per-sq-ft
    "low":    0.02,                  # updated from 0.05 to match Cell 9
    "medium": 0.05,                  # updated from 0.10
    "high":   0.10,                  # updated from 0.20
}

COLOR_IN_ZONE  = (0, 0, 255)    # BGR red   — person inside the monitored zone
COLOR_OUT_ZONE = (0, 255, 0)    # BGR green — person outside the zone


def _density_level(density: float, thresh: dict) -> str:
    if density < thresh["low"]:    return "LOW"
    if density < thresh["medium"]: return "MEDIUM"
    if density < thresh["high"]:   return "HIGH"
    return "CRITICAL"


def _build_homography(zone_pts: list, zone_sqft: float):
    """
    Derive a homography matrix from 4 zone pixel corners + known real-world area.

    Returns
    -------
    H_matrix  : np.ndarray  — 3×3 perspective transform (pixel → real-world ft)
    rw_poly   : Polygon     — shapely polygon in real-world coordinates
    """
    pts    = np.array(zone_pts, dtype=np.float32)
    cx, cy = pts.mean(axis=0)

    # Sort corners clockwise starting from top-left
    order = np.argsort(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    pts   = pts[order]
    sums  = pts[:, 0] + pts[:, 1]
    start = int(np.argmin(sums))
    pts   = np.roll(pts, -start, axis=0)
    tl, tr, br, bl = pts

    max_w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    max_h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))

    if max_w * max_h == 0:
        raise ValueError("Zone has zero pixel area — check corner points.")

    scale = np.sqrt(zone_sqft / (max_w * max_h))
    rw_w  = max_w * scale
    rw_h  = max_h * scale

    dst = np.array([[0, 0], [rw_w, 0], [rw_w, rw_h], [0, rw_h]], dtype=np.float32)
    src = np.array([tl, tr, br, bl], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Homography failed — try providing more precise corner points.")

    rw_poly = Polygon(dst.tolist())
    return H, rw_poly


def _point_to_realworld(px: float, py: float, H_mat: np.ndarray):
    """Project a single pixel foot-point into real-world coordinates."""
    pt     = np.array([[[px, py]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pt, H_mat)
    return float(warped[0][0][0]), float(warped[0][0][1])


@register_kpi
class DensityOccupancyKPI(BaseKPI):
    name         = "density_occupancy"
    display_name = "Density & Occupancy"

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
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
        min_box_w          = self._get("min_box_w",            _DEFAULT_MIN_BOX_W)
        min_box_h          = self._get("min_box_h",            _DEFAULT_MIN_BOX_H)
        dwell_min_frames   = self._get("dwell_min_frames",     _DEFAULT_DWELL_MIN_FRAMES)

        # zone_points: list of [x, y] pixel coords defining the ROI polygon.
        # Falls back to None → full-frame zone is used instead.
        zone_points_raw = self._get("zone_points", None)

        model = YOLO(model_path)
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── Build zone polygon & homography ───────────────────────────────────
        if zone_points_raw and len(zone_points_raw) >= 3:
            zone_pts = [tuple(p) for p in zone_points_raw]
        else:
            zone_pixel_pts = [(0, 0), (W, 0), (W, H), (0, H)]   # full frame

        # Attempt homography when exactly 4 corners are available (preferred).
        # For 3-point or irregular polygons, fall back to direct pixel containment.
        use_homography = len(zone_pixel_pts) == 4
        H_matrix       = None
        rw_zone_poly   = None
        zone_poly      = Polygon(zone_pixel_pts)   # pixel-space fallback

        if use_homography:
            try:
                H_matrix, rw_zone_poly = _build_homography(zone_pixel_pts, zone_sqft)
            except (ValueError, RuntimeError):
                # If homography setup fails, degrade gracefully to pixel containment
                use_homography = False

        # ── Tracking / alert state ────────────────────────────────────────────
        ids_inside_zone      = set()          # IDs currently stable-inside the zone
        ids_ever_seen        = set()          # cumulative unique IDs (foot traffic)
        consecutive_in_zone: dict[int, int]  = {}  # tid → frames continuously inside
        dwell_frames:        dict[int, int]  = {}  # tid → total in-zone frames
        consecutive_alert    = 0
        alert_active         = False
        total_footfall       = 0
        density              = 0.0
        occupancy_count      = 0
        d_level              = "LOW"

        frame_annotations: list[FrameAnnotation] = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self._observe(frame, frame_idx, job_id)

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf,
                iou=iou,
                classes=[0],        # person class only
                device=device,
                half=half,
                verbose=False,
            )

            if not results:
                frame_idx += 1
                continue

            result = results[0]

            detections:    list[Detection] = []
            status_lines:  list[str]       = []
            people_in_zone = 0

            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                xyxy_list = boxes.xyxy.int().cpu().tolist()
                confs     = boxes.conf.cpu().tolist()
                cls_list  = boxes.cls.int().cpu().tolist()

                for i, tid in enumerate(track_ids):
                    x1, y1, x2, y2 = xyxy_list[i]
                    conf_val = confs[i]
                    cls      = cls_list[i]

                    # ── Hard filter 1: must be class 0 (person) ────────────────
                    if cls != 0:
                        continue

                    # ── Hard filter 2: minimum bounding-box size ───────────────
                    # Rejects reflections, partial crops, and distant noise
                    if (x2 - x1) < min_box_w or (y2 - y1) < min_box_h:
                        continue

                    # Foot-point: bottom-centre of the bounding box
                    foot_x = int((x1 + x2) / 2)
                    foot_y = int(y2)

                    # ── Zone membership ────────────────────────────────────────
                    # Method A (preferred): homography → real-world containment
                    # Method B (fallback) : direct pixel-space containment
                    if use_homography and H_matrix is not None:
                        rw_x, rw_y = _point_to_realworld(foot_x, foot_y, H_matrix)
                        in_zone    = rw_zone_poly.contains(Point(rw_x, rw_y))
                    else:
                        in_zone = zone_poly.contains(Point(foot_x, foot_y))

                    # ── Dwell-gating ───────────────────────────────────────────
                    # Suppress single-frame ghost detections: a person must be
                    # seen inside the zone for ≥ dwell_min_frames consecutive
                    # frames before they are counted as occupying it.
                    if in_zone:
                        consecutive_in_zone[tid] = consecutive_in_zone.get(tid, 0) + 1
                    else:
                        consecutive_in_zone[tid] = 0

                    stable_in_zone = consecutive_in_zone.get(tid, 0) >= dwell_min_frames

                    if stable_in_zone:
                        people_in_zone           += 1
                        dwell_frames[tid]         = dwell_frames.get(tid, 0) + 1
                        ids_inside_zone.add(tid)
                    else:
                        # Only evict once they are fully outside (not just ungated)
                        if not in_zone:
                            ids_inside_zone.discard(tid)

                    # Footfall: every unique ID seen anywhere in the frame counts
                    ids_ever_seen.add(tid)

                    dwell_s    = dwell_frames.get(tid, 0) / fps
                    box_color  = COLOR_IN_ZONE if stable_in_zone else COLOR_OUT_ZONE
                    label      = f"ID:{tid} {conf_val:.2f} dwell:{dwell_s:.1f}s"

                    detections.append(
                        Detection(x1, y1, x2, y2, label, conf_val, color=box_color)
                    )

            # ── Metrics (computed every frame, outside the boxes guard) ────────
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
                    detections=list(detections),
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
                "device":             device,
                "used_homography":    use_homography,
            },
        )
