"""Staff Absence (catalog KPI #13) - the inverse of Guard Presence (#19),
sustained over time and gated by a shift schedule. Ported from
Research/staff_absence_prototype/ (absence.py/guard_presence.py/
schedule.py/zone_occupancy.py), reimplemented on the same
ultralytics-native track() + shapely stack as ..density_occupancy /
..occupancy_dwell rather than the prototype's separate `supervision` stack.

Per the prototype's own design: Guard Presence = OBJ(person) -> ZONE ->
SCHED - "is a person present in this zone, during the scheduled window".
Staff Absence is that same signal inverted into a timer: how long has the
post been continuously empty while a guard was actually expected there.

KNOWN LIMITATION (carried over from the prototype, see guard_presence.py's
docstring there): "guard present" is currently ANY confirmed-inside
person - a visitor or passer-by resets the absence timer just like a real
guard would. Swap `_present()` for a uniform/role classifier or ReID match
once that model exists; nothing else here needs to change.

The zone polygon is per-camera (see ..zone_labels.get_camera_zone_points,
drawn via POST /api/cameras/{camera_id}/labels); falls back to the full
frame if this camera has none saved yet.
"""
from datetime import datetime, time as dtime
from typing import Optional

import cv2
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ..zone_labels import get_camera_zone_points
from ...config import settings


def _schedule_active(window: Optional[str], now: Optional[datetime] = None) -> bool:
    """window: "HH:MM-HH:MM", or None/empty to be always active. Overnight
    windows (e.g. "22:00-06:00") are supported."""
    if not window:
        return True
    start_s, end_s = window.split("-")
    start, end = dtime.fromisoformat(start_s.strip()), dtime.fromisoformat(end_s.strip())
    t = (now or datetime.now()).time()
    if start <= end:
        return start <= t < end
    return t >= start or t < end   # overnight wrap


def _present(occupancy: int) -> bool:
    """occupancy: confirmed-inside person count. True if a guard is
    considered present at the post this frame - see module docstring's
    KNOWN LIMITATION."""
    return occupancy > 0


@register_kpi
class StaffAbsenceKPI(BaseKPI):
    name = "staff_absence"
    display_name = "Staff Absence"
    requires_zone = True

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path       = self._get("model_path",               "app/models/yolo26m.pt")
        conf              = self._get("confidence",                0.35)
        iou               = self._get("iou_threshold",             0.50)
        infer_imgsz       = self._get("infer_imgsz",                640)
        frame_stride      = max(1, int(self._get("frame_stride",     2)))
        persist_frames    = int(self._get("occupancy_persist_frames", 3))
        miss_grace        = int(self._get("miss_grace_frames",        5))
        absence_alert_secs = float(self._get("absence_alert_secs", 15.0))
        schedule_window   = self._get("schedule", None)   # "HH:MM-HH:MM" or None = always active
        alert_cooldown    = float(self._get("alert_cooldown_secs", 30.0))

        model = YOLO(model_path)
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        zone_points = get_camera_zone_points(job_id, self.name)
        zone_pts = [tuple(p) for p in zone_points] if zone_points and len(zone_points) >= 3 \
            else [(0, 0), (W, 0), (W, H), (0, H)]
        zone_poly = Polygon(zone_pts)

        state: dict[int, dict] = {}   # tid -> {streak, miss, confirmed}
        absent_since: Optional[float] = None
        last_alert_ts   = -1e9
        max_absent_secs = 0.0
        alert_events    = 0
        frame_idx       = 0

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
                            boxes_for_alert.append((x1, y1, x2, y2, "person", (0, 200, 60)))

                for tid in inside_ids:
                    st = state.setdefault(tid, {"streak": 0, "miss": 0, "confirmed": False})
                    st["streak"] += 1
                    st["miss"] = 0
                    if st["streak"] >= persist_frames:
                        st["confirmed"] = True

                for tid in list(state.keys()):
                    if tid in inside_ids:
                        continue
                    st = state[tid]
                    st["miss"] += 1
                    st["streak"] = 0
                    if st["miss"] > miss_grace:
                        del state[tid]

                occupancy = sum(
                    1 for tid, st in state.items()
                    if st["confirmed"] and (tid in inside_ids or st["miss"] <= miss_grace)
                )
                present = _present(occupancy)
                schedule_active = _schedule_active(schedule_window)

                if present or not schedule_active:
                    absent_since = None
                    absent_secs = 0.0
                else:
                    if absent_since is None:
                        absent_since = ts
                    absent_secs = ts - absent_since

                max_absent_secs = max(max_absent_secs, absent_secs)
                absence_alert = absence_alert_secs > 0 and absent_secs >= absence_alert_secs

                if absence_alert and (ts - last_alert_ts) > alert_cooldown:
                    last_alert_ts = ts
                    alert_events += 1
                    self._save_alert(
                        "staff_absence_detected", job_id, frame_idx, confidence=conf,
                        extra={"absent_secs": round(absent_secs, 1), "zone_polygon": zone_pts},
                        boxes=boxes_for_alert,
                    )

            frame_idx += 1

        cap.release()
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "alert_events":     alert_events,
            "max_absent_secs":  round(max_absent_secs, 1),
            "total_frames":     frame_idx,
            "device":           device,
        })
