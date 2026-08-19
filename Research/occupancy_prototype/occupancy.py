"""Occupancy (KPI #16) + per-person Dwell (KPI #27) over a polygon zone.

Live occupancy = number of tracked people currently inside the zone polygon.
Dwell        = how long each track has continuously been inside; alert past a threshold.

Two layers of noise-suppression, deliberately separate:
  * Membership debounce (occupancy_persist_frames): a track must be inside for N
    consecutive detect-frames before it counts, and may briefly disappear
    (miss_grace frames) without losing its dwell timer — kills boundary flicker and
    short detection dropouts. Lives here.
  * Alert debounce (EventBus persist+cooldown): applied by the runner on top of the
    dwell/overcrowd flags this module raises.

Anchor choice matters: BOTTOM_CENTER (feet) for body mode / oblique so a person is
"inside" when their feet are; CENTER for head mode / top-down. Set per zone in config.
"""
import numpy as np
import supervision as sv

_ANCHORS = {
    "bottom_center": sv.Position.BOTTOM_CENTER,
    "center":        sv.Position.CENTER,
    "top_center":    sv.Position.TOP_CENTER,
    "center_of_mass": sv.Position.CENTER_OF_MASS,
}


class _TrackState:
    __slots__ = ("streak", "miss", "enter_ts", "confirmed")

    def __init__(self):
        self.streak = 0
        self.miss = 0
        self.enter_ts = None
        self.confirmed = False


class OccupancyZone:
    def __init__(self, zone_cfg, persist_frames=3, miss_grace=5):
        self.name = zone_cfg["name"]
        polygon = np.array(zone_cfg["polygon"], dtype=int)
        if polygon.size < 6:  # need >= 3 points
            raise ValueError(
                f"zone '{self.name}' has no polygon — draw one with zone_drawer.py "
                f"and paste the points into config.yaml")
        anchor = _ANCHORS.get(zone_cfg.get("anchor", "bottom_center"),
                              sv.Position.BOTTOM_CENTER)
        self.zone = sv.PolygonZone(polygon=polygon, triggering_anchors=(anchor,))
        self.polygon = polygon

        self.dwell_alert_secs = float(zone_cfg.get("dwell_alert_secs", 0.0))
        self.occupancy_alert = int(zone_cfg.get("occupancy_alert", 0))
        self.persist = int(persist_frames)
        self.miss_grace = int(miss_grace)
        self._state = {}   # tracker_id -> _TrackState

    def update(self, detections, ts):
        """detections: sv.Detections carrying tracker_id. Returns a dict:
            occupancy       : int   live confirmed-inside count
            people          : list[(track_id, dwell_secs)] for confirmed-inside tracks
            dwell_alerts    : list[(track_id, dwell_secs)] over threshold this frame
            overcrowd       : bool  occupancy above configured limit
        """
        # Which tracked ids are geometrically inside this frame.
        inside_ids = set()
        if len(detections) > 0 and detections.tracker_id is not None:
            mask = self.zone.trigger(detections)
            for tid, ins in zip(detections.tracker_id, mask):
                if ins and tid is not None:
                    inside_ids.add(int(tid))

        # Advance streaks for tracks inside this frame.
        for tid in inside_ids:
            st = self._state.setdefault(tid, _TrackState())
            st.streak += 1
            st.miss = 0
            if st.streak >= self.persist:
                st.confirmed = True
                if st.enter_ts is None:
                    st.enter_ts = ts

        # Age tracks that were being followed but are not inside this frame.
        for tid in list(self._state.keys()):
            if tid in inside_ids:
                continue
            st = self._state[tid]
            st.miss += 1
            st.streak = 0
            if st.miss > self.miss_grace:
                del self._state[tid]   # left the zone -> dwell resets

        # Report on confirmed tracks that are present (inside now, or within grace).
        people, dwell_alerts = [], []
        for tid, st in self._state.items():
            if not st.confirmed or st.enter_ts is None:
                continue
            present = tid in inside_ids or st.miss <= self.miss_grace
            if not present:
                continue
            dwell = ts - st.enter_ts
            people.append((tid, dwell))
            if self.dwell_alert_secs > 0 and dwell >= self.dwell_alert_secs:
                dwell_alerts.append((tid, dwell))

        occupancy = len(people)
        overcrowd = self.occupancy_alert > 0 and occupancy > self.occupancy_alert
        return {
            "occupancy": occupancy,
            "people": people,
            "dwell_alerts": dwell_alerts,
            "overcrowd": overcrowd,
        }
