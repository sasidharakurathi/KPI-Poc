"""In-process sliding-window rate limiter.

No external infra (Redis/etc.) — adequate for a single-process deployment,
which is what this app runs as today. If this is ever scaled to multiple
worker processes, move this state to a shared store (redis is already an
unused dependency sitting in the venv).
"""
import threading
import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, window_seconds: float) -> int:
        """Record a hit for `key` and return the number of hits still inside
        the trailing `window_seconds` window (including this one)."""
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            dq = self._hits[key]
            dq.append(now)
            while dq and dq[0] < cutoff:
                dq.popleft()
            return len(dq)

    def reset(self) -> None:
        """Clear all recorded hits — used by tests, which otherwise share this
        process-global state across test cases."""
        with self._lock:
            self._hits.clear()


# Shared limiters for Phase 0 auth endpoints (PRD §2.3: rate-limited per IP
# and per username on login and password-reset).
login_limiter = SlidingWindowLimiter()
password_reset_limiter = SlidingWindowLimiter()
