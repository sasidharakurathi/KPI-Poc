"""Vehicle checkpoint compliance: does every vehicle that enters the drawn
zone come to a genuine stop (a security/inspection check) before it leaves,
or does it drive straight through? Speed itself is not computed - the
previous version's calibration was a hardcoded placeholder (a well-known
tutorial's example points) never matched to a real camera, so any km/h it
reported was meaningless. Per user decision, dropped entirely rather than
kept half-working; detection + tracking + zone stillness is what matters.

Detection model: the original `vehicle-detection-speed.pt` (a 6MB custom
checkpoint with ONLY 5 vehicle classes, no "person") was replaced with the
standard COCO-trained `yolo26m.pt` already used elsewhere in this project
(..occupancy_dwell, ..people_count), filtered to just the vehicle classes
(car/motorcycle/bus/truck). Two concrete problems on our own footage drove
this: the old model confidently labeled a walking person "Motorcycle" (it
has no way to say "not a vehicle" - it must pick one of its 5 labels), and
a car's raw detection confidence fluctuated well below the production conf
threshold for extended stretches, dropping its track. COCO's `person` class
lets the model correctly reject non-vehicles instead of guessing, and its
far larger/more diverse training set gave more frames above threshold in
the same dip window when tested directly against this footage.

A vehicle counts as "in the zone" whenever its detection box overlaps the
drawn polygon at all (not a ground-contact-point convention like the other
zone KPIs use for pedestrians - the user asked for a plain "is the vehicle
present in the drawn box" check instead).

The rule is deliberately simple (user's call, after movement/stillness-based
attempts proved too fragile to tune reliably): **a vehicle that stays in the
zone for at least `checked_dwell_seconds` (default 20s, configurable) counts
as checked.** No stillness/velocity measurement at all - dwell time only.

The verdict is DEFERRED: while a vehicle is in the zone nothing is decided or
reported, it is only timed. The checked/not-checked verdict is issued once,
at the moment the vehicle is confirmed to have fully left the zone.

Dwell is measured as ELAPSED TIME between the vehicle's first and last
sighting in the zone, NOT as a count of consecutive detections. That matters:
measured on real footage, this model loses a stationary car for up to 2.4s at
a stretch (its confidence on distant vehicles swings between ~0.06 and ~0.96),
so a consecutive-frames count would reset constantly and a genuinely parked
car would never accumulate its dwell. For the same reason
`zone_miss_grace_frames` must stay comfortably above the worst observed
detection gap, or one continuous stop gets chopped into several short visits,
each too brief to qualify.

Each tracker id is only ever given ONE verdict - a vehicle straddling the zone
edge and flickering in/out within `zone_miss_grace_frames` cannot produce
repeated alerts for the same pass. A track id that switches mid-visit (the
detector drops the vehicle, ByteTrack hands back a new id when it reappears)
is bridged onto the existing visit by POSITION, but only while that visit is
still active - i.e. within `zone_miss_grace_frames` (~3.6s at current config)
of its last sighting. That bound is deliberate and load-bearing: an earlier,
unbounded version of this re-link matched against a much longer "recently
parked" window (to try to bridge a ~14s trunk-inspection blackout) and, on
real footage, fused three DIFFERENT vehicles that queued in the same spot
13.9-15.2s apart into one fake visit - a worse failure (a swallowed verdict)
than under-counting. Capping the match to the existing miss-grace window
keeps a >3x margin below that measured 13.9s different-vehicle gap while
still fixing the case that motivated bringing re-linking back: a person
briefly occluding a vehicle (a fraction of a second up to a few seconds)
splits its detection into two-or-more raw track ids for one physical vehicle,
confirmed on real footage by the id hand-off happening at the same or an
adjacent timestamp, not a long gap. A brand-new raw id is bridged onto the
nearest still-active visit only if it's within `RELINK_MAX_DIST_PX` of that
visit's last box - guards against two different vehicles genuinely queued
together in the same small zone at once.

A vehicle still inside the zone when the video/clip ends is left unresolved
(no verdict either way) rather than guessed at.

Zone: per-camera drawn polygon (BaseKPI.requires_zone, see ..zone_labels),
falls back to a `zone_points` config value, else the whole frame.
"""
import tempfile
from pathlib import Path

import cv2
from shapely.geometry import Polygon, box
from ultralytics import YOLO

from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ..zone_labels import get_camera_zone_points
from ...config import settings

# COCO class ids for the vehicle-detection-speed model's own former classes
VEHICLE_CLASS_IDS = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}


def _box_center(b):
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2, (y1 + y2) / 2


def _center_dist(b1, b2):
    (cx1, cy1), (cx2, cy2) = _box_center(b1), _box_center(b2)
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def _write_tracker_config(track_buffer: int, high_thresh: float, low_thresh: float,
                          new_thresh: float, match_thresh: float) -> str:
    """ByteTrack config with a longer-lived lost-track buffer than the stock
    one. Ultralytics' default `track_buffer: 30` works out to only ~2.4s at
    this pipeline's tracker input rate (source fps / frame_stride), and this
    model loses a stationary vehicle for up to ~2.4s at a time, so a parked
    car kept being re-IDed mid-stop and its dwell restarted from zero.
    Written per run so the values stay config-driven rather than a checked-in
    file nobody remembers to edit."""
    cfg = (
        "tracker_type: bytetrack\n"
        f"track_high_thresh: {high_thresh}\n"
        f"track_low_thresh: {low_thresh}\n"
        f"new_track_thresh: {new_thresh}\n"
        f"track_buffer: {track_buffer}\n"
        f"match_thresh: {match_thresh}\n"
        "fuse_score: True\n"
    )
    fh = tempfile.NamedTemporaryFile("w", suffix="_bytetrack.yaml", delete=False, encoding="utf-8")
    fh.write(cfg)
    fh.close()
    return fh.name


@register_kpi
class SpeedTrackerKPI(BaseKPI):
    name = "speed_tracker"
    display_name = "Vehicle Checkpoint Compliance"
    requires_zone = True

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path     = self._get("model_path",                 "app/models/yolo26m.pt")
        conf           = self._get("confidence",                 0.50)
        infer_imgsz    = int(self._get("infer_imgsz",              640))
        frame_stride   = max(1, int(self._get("frame_stride",        2)))
        persist_frames = max(1, int(self._get("zone_persist_frames", 2)))
        miss_grace     = int(self._get("zone_miss_grace_frames",     45))
        checked_dwell  = float(self._get("checked_dwell_seconds",  20.0))
        min_verdict    = float(self._get("min_verdict_dwell_seconds", 1.5))
        track_buffer   = int(self._get("track_buffer",              150))
        track_high     = float(self._get("track_high_thresh",      0.25))
        track_low      = float(self._get("track_low_thresh",        0.1))
        new_track      = float(self._get("new_track_thresh",       0.25))
        match_thresh   = float(self._get("track_match_thresh",      0.8))

        tracker_cfg = _write_tracker_config(track_buffer, track_high, track_low, new_track, match_thresh)

        model = YOLO(model_path)
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        zone_points = get_camera_zone_points(job_id, self.name) or self._get("zone_points")
        zone_pts = [tuple(p) for p in zone_points] if zone_points and len(zone_points) >= 3 \
            else [(0, 0), (W, 0), (W, H), (0, H)]
        zone_poly = Polygon(zone_pts)

        # Position re-link radius: generous relative to the zone itself (see
        # module docstring) so a vehicle's box can drift/grow between
        # fragments, but still small enough that two genuinely different
        # vehicles queued in the same tiny zone won't be mistaken for one.
        zminx, zminy, zmaxx, zmaxy = zone_poly.bounds
        zone_diag = ((zmaxx - zminx) ** 2 + (zmaxy - zminy) ** 2) ** 0.5
        relink_max_dist = float(self._get("relink_max_dist_px", max(zone_diag * 3, 150.0)))

        # canonical_tid -> {sightings, miss, first_t, last_t, class_name, last_box}
        state: dict[int, dict] = {}
        # raw ByteTrack tid -> canonical tid it's been re-linked onto (see
        # module docstring for why this is safely bounded to miss_grace)
        alias: dict[int, int] = {}
        vehicles_checked = 0
        alert_events = 0
        frame_idx = 0

        def _issue_verdict(tid, st, at_frame):
            nonlocal vehicles_checked, alert_events
            dwell = st["last_t"] - st["first_t"]
            if st["sightings"] < persist_frames or dwell < min_verdict:
                return
            if dwell >= checked_dwell:
                vehicles_checked += 1
            else:
                alert_events += 1
                x1, y1, x2, y2 = st["last_box"]
                self._save_alert(
                    "vehicle_no_check", job_id, at_frame,
                    confidence=conf,
                    extra={"track_id": int(tid),
                           "class_name": st["class_name"],
                           "dwell_seconds": round(dwell, 1),
                           "required_seconds": checked_dwell},
                    boxes=[(x1, y1, x2, y2, f"#{tid} NO CHECK", (0, 0, 255))],
                )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self._observe(frame, frame_idx, job_id)

            if frame_idx % frame_stride == 0:
                # agnostic_nms: one vehicle must not survive NMS twice as both a
                # "Car" and a "Truck" box - class-wise NMS let that through, and
                # each duplicate got its own track id
                results = model.track(
                    frame, persist=True, tracker=tracker_cfg,
                    conf=conf, imgsz=infer_imgsz, classes=list(VEHICLE_CLASS_IDS),
                    agnostic_nms=True, device=device, half=half, verbose=False,
                )

                inside_ids: set[int] = set()
                if results and results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes
                    track_ids = boxes.id.int().cpu().tolist()
                    cls_ids   = boxes.cls.int().cpu().tolist()
                    xyxy_list = boxes.xyxy.cpu().tolist()

                    ts = frame_idx / fps
                    for tid, cls_id, (x1, y1, x2, y2) in zip(track_ids, cls_ids, xyxy_list):
                        if not zone_poly.intersects(box(x1, y1, x2, y2)):
                            continue

                        canonical = alias.get(tid)
                        if canonical is None and tid not in state:
                            # brand-new raw id: bridge onto the nearest still
                            # -active visit (miss_grace hasn't expired it yet)
                            # if it's close enough to be the same vehicle
                            best_tid, best_dist = None, None
                            for other_tid, other_st in state.items():
                                if other_st["last_box"] is None:
                                    continue
                                d = _center_dist(other_st["last_box"], (x1, y1, x2, y2))
                                if d <= relink_max_dist and (best_dist is None or d < best_dist):
                                    best_tid, best_dist = other_tid, d
                            if best_tid is not None:
                                canonical = best_tid
                                alias[tid] = canonical
                        if canonical is None:
                            canonical = tid

                        inside_ids.add(canonical)
                        st = state.setdefault(canonical, {
                            "sightings": 0, "miss": 0, "first_t": ts,
                            "class_name": VEHICLE_CLASS_IDS.get(cls_id, "vehicle"),
                            "last_box": None,
                        })
                        st["sightings"] += 1
                        st["miss"] = 0
                        st["last_t"] = ts
                        st["last_box"] = (int(x1), int(y1), int(x2), int(y2))

                for tid in list(state.keys()):
                    if tid in inside_ids:
                        continue
                    st = state[tid]
                    st["miss"] += 1
                    if st["miss"] > miss_grace:
                        _issue_verdict(tid, st, frame_idx)
                        del state[tid]

            frame_idx += 1

        # anything still in the zone at end of video is left unresolved

        cap.release()
        Path(tracker_cfg).unlink(missing_ok=True)
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "alert_events":     alert_events,
            "vehicles_checked": vehicles_checked,
            "total_frames":     frame_idx,
            "device":           device,
        })
