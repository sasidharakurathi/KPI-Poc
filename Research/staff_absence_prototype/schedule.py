"""SCHED primitive — shift-window gate for schedule-based KPIs.

PROTOTYPE_BUILD_GUIDE.md's checklist flags this explicitly: "Schedule-based KPIs (staff
absence, guard tour) use the correct timezone." A post isn't "absent" if no guard is expected
there in the first place, so absence.py only accumulates/alerts while the schedule is active.

Uses local wall-clock time (datetime.now()), matching how a live deployment actually runs —
not the video timestamp, which only means something for a real-time source anyway. For offline
test clips, leave the zone's `schedule` unset (None = always active) unless you specifically
want to test the gating behavior.
"""
from datetime import datetime, time as dtime


class Schedule:
    def __init__(self, window: str | None):
        """window: "HH:MM-HH:MM", or None/empty to be always active.
        Overnight windows (e.g. "22:00-06:00") are supported."""
        self.window = window
        self._start = None
        self._end = None
        if window:
            start_s, end_s = window.split("-")
            self._start = dtime.fromisoformat(start_s.strip())
            self._end = dtime.fromisoformat(end_s.strip())

    def is_active(self, dt: datetime | None = None) -> bool:
        if self._start is None:
            return True
        now = (dt or datetime.now()).time()
        if self._start <= self._end:
            return self._start <= now < self._end
        return now >= self._start or now < self._end   # overnight wrap
