"""Adaptive per-camera throughput controller for the IP-camera clip pipeline.

Goal: the *combined* time to run every KPI assigned to a camera over one
recorded clip should stay safely under that clip's own duration (currently
STREAM_CLIP_SECONDS, e.g. 60s) — otherwise the clip_processor queue falls
further behind the live recording rate with every clip.

Technique: AIMD feedback control (the same family as TCP congestion control)
— one numeric "skip multiplier" per camera, updated after every clip:
  - total wall time over budget      -> multiply the skip factor UP
    (every KPI on that camera processes fewer frames on the next clip)
  - comfortably under budget         -> ease the skip factor back DOWN
    towards 1.0 (each KPI's own configured baseline), recovering accuracy
    now that there's compute headroom to spare.
The multiplier is applied on top of each KPI's own configured frame_stride
(see BaseKPI._effective_stride) — it can only make sampling *coarser* than
that baseline, never finer, since the baseline reflects the minimum
granularity a KPI actually needs to detect its target behavior.

This alone is "dumb" uniform frame-skipping. BaseKPI._should_process_frame
pairs it with a cheap motion-diff override (see that method) so a
stride-skipped frame is still processed if the scene changed enough since
the last processed frame — the adaptive part decides *how much* to thin out
static content, the motion gate protects against silently dropping a brief
but real event just because the stride said to skip it.
"""
import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_MIN_MULTIPLIER = 1.0
_MAX_MULTIPLIER = 6.0
_GROWTH_FACTOR   = 1.35   # multiplicative increase when over budget
_DECAY_FACTOR    = 0.92   # gentle decrease when comfortably under budget

# "Budget" is a fraction of the clip's own duration, not the whole thing —
# leaves headroom for jitter (disk I/O, model warmup on first use, etc.) so
# the queue doesn't creep up even when running right at the edge.
_TARGET_UTILIZATION = 0.80
# Only ease the multiplier back down when usage drops comfortably below
# budget (not just under it) — avoids oscillating stride every single clip.
_EASE_THRESHOLD = 0.55


@dataclass
class _CameraState:
    multiplier: float = 1.0
    last_ratio: float = 0.0   # wall_time / clip_duration of the most recent clip


class AdaptiveController:
    def __init__(self) -> None:
        self._state: dict[str, _CameraState] = {}
        self._lock = threading.Lock()

    def get_multiplier(self, camera_id: str) -> float:
        with self._lock:
            st = self._state.get(camera_id)
            return st.multiplier if st else _MIN_MULTIPLIER

    def record_clip_result(self, camera_id: str, clip_duration_sec: float, wall_time_sec: float) -> None:
        """Call once per clip with the TOTAL wall-clock time across every KPI
        that ran on it (not per-KPI) — the budget is a combined one."""
        if not camera_id or clip_duration_sec <= 0:
            return
        ratio = wall_time_sec / clip_duration_sec

        with self._lock:
            st = self._state.setdefault(camera_id, _CameraState())
            prev = st.multiplier

            if ratio > _TARGET_UTILIZATION:
                st.multiplier = min(_MAX_MULTIPLIER, st.multiplier * _GROWTH_FACTOR)
            elif ratio < _EASE_THRESHOLD:
                st.multiplier = max(_MIN_MULTIPLIER, st.multiplier * _DECAY_FACTOR)

            st.last_ratio = ratio
            changed = abs(st.multiplier - prev) > 1e-6

        level = logger.info if changed else logger.debug
        level(
            f"[adaptive] {camera_id}: {wall_time_sec:.1f}s / {clip_duration_sec:.1f}s clip "
            f"(utilization={ratio:.2f}) — skip_multiplier {prev:.2f} -> {st.multiplier:.2f}"
        )

    def status_for(self, camera_id: str) -> dict:
        with self._lock:
            st = self._state.get(camera_id)
        if not st:
            return {"skip_multiplier": _MIN_MULTIPLIER, "last_utilization": None}
        return {"skip_multiplier": round(st.multiplier, 2), "last_utilization": round(st.last_ratio, 2)}


adaptive_controller = AdaptiveController()
