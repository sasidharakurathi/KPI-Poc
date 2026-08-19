"""Debounce + cooldown so an alert fires only on sustained evidence, not single frames.
Key is typically (camera, kpi_name, zone, track_id).

gap_tol: consecutive inactive frames tolerated before the streak resets.
A single noisy YOLO frame (bbox briefly off, detection dropout) should not kill
a streak that has been building for many frames.

Copied verbatim from pose_prototype/event_layer.py so the occupancy prototype shares
the exact same, already-validated debounce semantics as the pose KPIs.
"""
from collections import defaultdict


class EventBus:
    def __init__(self, persist_frames=3, cooldown_secs=8.0, gap_tol=2):
        self.persist  = persist_frames
        self.cooldown = cooldown_secs
        self.gap_tol  = gap_tol
        self.streak   = defaultdict(int)
        self.gap      = defaultdict(int)
        self.last     = defaultdict(lambda: -1e9)

    def submit(self, key, active, ts):
        if active:
            self.streak[key] += 1
            self.gap[key]     = 0
        else:
            self.gap[key] += 1
            if self.gap[key] > self.gap_tol:
                self.streak[key] = 0
        if self.streak[key] >= self.persist and (ts - self.last[key]) > self.cooldown:
            self.last[key] = ts
            return True
        return False
