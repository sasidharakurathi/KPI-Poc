"""Staff Absence — inverse of Guard Presence (KPI #19), sustained over time and gated by SCHED.

Per KPI_Pipeline_Architecture.md: Guard Presence = OBJ(person) -> ZONE -> SCHED. Staff Absence is
that same signal, inverted into a timer: how long has the post been continuously empty (while a
guard was actually expected there)? Composes the already-validated pieces instead of re-deriving
zone/track logic:
  zone_occupancy.OccupancyZone  -> confirmed-inside person count (membership debounce + miss_grace)
  guard_presence.detect_guard_presence -> the swappable presence predicate
  schedule.Schedule             -> only count absence during the guard's expected shift

Two layers of noise suppression carry over unchanged from occupancy_prototype:
  * Membership debounce (inside OccupancyZone) — kills zone-boundary flicker on the person side.
  * Alert debounce (EventBus, applied by the runner) — requires the absence_alert flag to persist
    before actually firing an [EVENT], and enforces a cooldown between repeats.
This module's own absent_secs timer is a THIRD, deliberate layer: the post must be empty for
absence_alert_secs of continuous (schedule-active) time before absence_alert even turns True.
"""
from zone_occupancy import OccupancyZone
from guard_presence import detect_guard_presence
from schedule import Schedule


class StaffAbsenceZone:
    def __init__(self, zone_cfg, persist_frames=3, miss_grace=5):
        self.name = zone_cfg["name"]
        self._occ = OccupancyZone(zone_cfg, persist_frames=persist_frames, miss_grace=miss_grace)
        self.polygon = self._occ.polygon
        self.absence_alert_secs = float(zone_cfg.get("absence_alert_secs", 0.0))
        self.schedule = Schedule(zone_cfg.get("schedule"))
        self._absent_since = None   # ts when the zone first went empty during an active shift

    def update(self, detections, ts):
        """detections: sv.Detections carrying tracker_id. Returns a dict:
            present         : bool   guard presence this frame (post detect_guard_presence)
            occupancy       : int    confirmed-inside person count
            absent_secs     : float  continuous empty duration during the active shift
            absence_alert   : bool   absent_secs past the configured threshold
            schedule_active : bool   whether a guard is currently expected at this post
        """
        res = self._occ.update(detections, ts)
        occupancy = res["occupancy"]
        present = detect_guard_presence(occupancy)
        schedule_active = self.schedule.is_active()

        if present or not schedule_active:
            self._absent_since = None
            absent_secs = 0.0
        else:
            if self._absent_since is None:
                self._absent_since = ts
            absent_secs = ts - self._absent_since

        absence_alert = (self.absence_alert_secs > 0 and
                          absent_secs >= self.absence_alert_secs)

        return {
            "present": present,
            "occupancy": occupancy,
            "absent_secs": absent_secs,
            "absence_alert": absence_alert,
            "schedule_active": schedule_active,
        }
