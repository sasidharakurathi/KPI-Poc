"""EventBus -- per-key streak/gap/cooldown debounce; submit() returns True once a streak is confirmed."""
from typing import Any


class EventBus:
    def __init__(
        self,
        persist_frames: int = 3,
        cooldown_secs: float = 8.0,
        gap_tol: int = 2,
        fps: float = 25.0,
    ) -> None:
        self._persist  = max(1, persist_frames)
        self._cooldown = max(0, int(cooldown_secs * fps))
        self._gap_tol  = max(0, gap_tol)
        self._state: dict[Any, dict] = {}

    def _get(self, key: Any) -> dict:
        if key not in self._state:
            self._state[key] = {"streak": 0, "gap": 0, "cd": 0}
        return self._state[key]

    def submit(self, key: Any, is_active: bool, _frame_idx: int = 0) -> bool:
        """Return True exactly once when a confirmed streak is reached."""
        s = self._get(key)

        if s["cd"] > 0:
            s["cd"] -= 1
            return False

        if is_active:
            s["streak"] += 1
            s["gap"] = 0
        else:
            s["gap"] += 1
            if s["gap"] > self._gap_tol:
                s["streak"] = 0
                s["gap"]    = 0

        if s["streak"] >= self._persist:
            s["cd"]     = self._cooldown
            s["streak"] = 0
            s["gap"]    = 0
            return True

        return False

    def reset(self, key: Any) -> None:
        self._state.pop(key, None)
