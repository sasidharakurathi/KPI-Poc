"""Occupancy + per-person Dwell Time over a drawn zone polygon (catalog KPI
#27, "Occupancy Count & Dwell Time"). Ported from
Research/occupancy_prototype/ (detector.py/tracker.py/occupancy.py), but
reimplemented on the same ultralytics-native track() + shapely stack as
..density_occupancy rather than the prototype's separate `supervision`
(ByteTrack/PolygonZone) composition, so this KPI matches the rest of
app/kpis/ instead of introducing a second tracking stack.

Two layers of noise suppression, same as the prototype:
  * Membership debounce (occupancy_persist_frames/miss_grace_frames): a
    track must be inside the zone for N consecutive processed frames
    before it counts, and may miss up to miss_grace_frames without its
    dwell timer resetting - kills boundary flicker and short dropouts.
  * Alert cooldown (alert_cooldown_secs): once fired, the same alert type
    won't fire again until the cooldown elapses.

The zone polygon is per-camera (see ..zone_labels.get_camera_zone_points,
drawn via POST /api/cameras/{camera_id}/labels); falls back to the full
frame if this camera has none saved yet.
"""
import cv2
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ..zone_labels import get_camera_zone_points
from ...config import settings


@register_kpi
class OccupancyDwellKPI(BaseKPI):
    name = "occupancy_dwell"
    display_name = "Occupancy Count & Dwell Time"
    requires_zone = True

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path        = self._get("model_path",               "app/models/yolo26m.pt")
        conf               = self._get("confidence",                0.35)
        iou                = self._get("iou_threshold",             0.50)
        infer_imgsz        = self._get("infer_imgsz",                640)
        frame_stride       = max(1, int(self._get("frame_stride",     2)))
        persist_frames     = int(self._get("occupancy_persist_frames", 3))
        miss_grace         = int(self._get("miss_grace_frames",        5))
        dwell_alert_secs   = float(self._get("dwell_alert_secs",     8.0))
        occupancy_alert    = int(self._get("occupancy_alert",          0))
        alert_cooldown     = float(self._get("alert_cooldown_secs", 30.0))

        model = YOLO(model_path)
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        zone_points = get_camera_zone_points(job_id, self.name)
        zone_pts = [tuple(p) for p in zone_points] if zone_points and len(zone_points) >= 3 \
            else [(0, 0), (W, 0), (W, H), (0, H)]
        zone_poly = Polygon(zone_pts)

        state: dict[int, dict] = {}   # tid -> {streak, miss, enter_ts, confirmed}
        last_alert_ts  = -1e9
        max_occupancy  = 0
        alert_events   = 0
        frame_idx      = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self._observe(frame, frame_idx, job_id)

            if frame_idx % frame_stride == 0:
                ts = frame_idx / fps
                results = model.track(
                    frame, persist=True, tracker="bytetrack.yaml",
                    conf=conf, iou=iou, imgsz=infer_imgsz, classes=[0],
                    device=device, half=half, verbose=False,
                )

                inside_ids: set[int] = set()
                boxes_for_alert = []
                if results and results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes
                    track_ids = boxes.id.int().cpu().tolist()
                    xyxy_list = boxes.xyxy.cpu().tolist()
                    for tid, (x1, y1, x2, y2) in zip(track_ids, xyxy_list):
                        foot = Point((x1 + x2) / 2, y2)
                        if zone_poly.contains(foot):
                            inside_ids.add(tid)
                            boxes_for_alert.append((x1, y1, x2, y2, f"#{tid}", (0, 200, 60)))

                for tid in inside_ids:
                    st = state.setdefault(tid, {"streak": 0, "miss": 0, "enter_ts": None, "confirmed": False})
                    st["streak"] += 1
                    st["miss"] = 0
                    if st["streak"] >= persist_frames:
                        st["confirmed"] = True
                        if st["enter_ts"] is None:
                            st["enter_ts"] = ts

                for tid in list(state.keys()):
                    if tid in inside_ids:
                        continue
                    st = state[tid]
                    st["miss"] += 1
                    st["streak"] = 0
                    if st["miss"] > miss_grace:
                        del state[tid]   # left the zone -> dwell resets

                occupancy = 0
                dwell_alert = None
                for tid, st in state.items():
                    if not st["confirmed"] or st["enter_ts"] is None:
                        continue
                    if tid not in inside_ids and st["miss"] > miss_grace:
                        continue
                    occupancy += 1
                    dwell = ts - st["enter_ts"]
                    if dwell_alert_secs > 0 and dwell >= dwell_alert_secs and dwell_alert is None:
                        dwell_alert = (tid, dwell)

                max_occupancy = max(max_occupancy, occupancy)
                overcrowd = occupancy_alert > 0 and occupancy > occupancy_alert

                if (overcrowd or dwell_alert) and (ts - last_alert_ts) > alert_cooldown:
                    last_alert_ts = ts
                    alert_events += 1
                    extra = {"occupancy": occupancy, "zone_polygon": zone_pts}
                    if dwell_alert:
                        extra["dwell_track_id"], extra["dwell_secs"] = dwell_alert[0], round(dwell_alert[1], 1)
                    self._save_alert(
                        "occupancy_overcrowd" if overcrowd else "occupancy_dwell_exceeded",
                        job_id, frame_idx, confidence=conf, extra=extra, boxes=boxes_for_alert,
                    )

            frame_idx += 1

        cap.release()
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "alert_events":  alert_events,
            "max_occupancy": max_occupancy,
            "total_frames":  frame_idx,
            "device":        device,
        })
