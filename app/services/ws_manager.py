"""In-process real-time pub-sub for the alerts WebSocket — Phase 4.

No new infrastructure (no Redis/broker) — a single-process asyncio connection
registry is sufficient for this deployment shape (one backend process per
site, matching the rest of this platform's single-org-per-deployment design).

Two call surfaces:
  - `broadcast()` (async) — call from code already running on the main event
    loop (e.g. the WebSocket endpoint itself, or an async background task).
  - `broadcast_threadsafe()` (sync) — call from a background thread that is
    NOT on the main event loop, e.g. the synchronous video-detection pipeline
    (app.kpis.base._flush -> app.db.create_alert). Schedules the same
    broadcast onto the main loop via asyncio.run_coroutine_threadsafe instead
    of running it inline.
"""
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class _Connection:
    """One live WebSocket connection plus the zone restriction it was
    authenticated with at connect time (None = unrestricted, sees everything)."""

    __slots__ = ("websocket", "allowed_camera_ids")

    def __init__(self, websocket, allowed_camera_ids: Optional[set[str]]) -> None:
        self.websocket = websocket
        self.allowed_camera_ids = allowed_camera_ids

    def can_see(self, camera_id: Optional[str]) -> bool:
        """None (unrestricted) sees every event, including camera-less ones.
        A zone-restricted connection only sees events tied to a camera in its
        allowed set — a camera-less event (e.g. an alert from an ad-hoc video
        upload with no registered camera) is invisible to it, matching the
        same "hide camera-less alerts from zone-restricted roles" rule
        applied to the REST endpoints."""
        if self.allowed_camera_ids is None:
            return True
        return camera_id is not None and camera_id in self.allowed_camera_ids


class AlertsWebSocketManager:
    def __init__(self) -> None:
        self._connections: list[_Connection] = []
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once at app startup, from the main event loop — lets
        broadcast_threadsafe() schedule work onto it safely from any other
        thread."""
        self._loop = loop

    async def connect(self, websocket, allowed_camera_ids: Optional[set[str]]) -> _Connection:
        conn = _Connection(websocket, allowed_camera_ids)
        async with self._lock:
            self._connections.append(conn)
        return conn

    async def disconnect(self, conn: _Connection) -> None:
        async with self._lock:
            if conn in self._connections:
                self._connections.remove(conn)

    async def broadcast(self, event_type: str, payload: dict, camera_id: Optional[str] = None) -> None:
        """Sends {"event": event_type, "data": payload} to every connection
        whose zone restriction allows it to see this camera_id."""
        message = {"event": event_type, "data": payload}
        async with self._lock:
            targets = [c for c in self._connections if c.can_see(camera_id)]

        stale: list[_Connection] = []
        for conn in targets:
            try:
                await conn.websocket.send_json(message)
            except Exception:
                stale.append(conn)
        for conn in stale:
            await self.disconnect(conn)

    def broadcast_threadsafe(self, event_type: str, payload: dict, camera_id: Optional[str] = None) -> None:
        """Safe to call from any thread, including the synchronous video
        pipeline's worker thread. Best-effort: if the websocket layer hasn't
        started (e.g. a script importing app.db outside the running FastAPI
        app), this is a silent no-op rather than an error — broadcasting is
        never allowed to break the actual alert-creation path."""
        if self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self.broadcast(event_type, payload, camera_id), self._loop
            )
        except Exception:
            logger.exception("[ws_manager] failed to schedule broadcast of '%s'", event_type)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


ws_manager = AlertsWebSocketManager()
