"""Camera-offline heartbeat monitor — Phase 4.

Periodically compares each recording-enabled camera's live stream status
(app.stream_recorder.stream_recorder_manager) against its last-known
Camera.connectivity_status, persists any transition, and on Active->Inactive
creates a "Camera Offline" Alert (job_id=None — there's no video-processing
job behind a connectivity signal) plus broadcasts camera.offline /
camera.online events over the realtime websocket.

Audit-log wiring is deliberately not included here — it depends on Phase 8's
audit_service, which doesn't exist yet.

`_check_once()` is a plain synchronous function (all it does is DB reads/
writes and a threadsafe broadcast call) so it's directly unit-testable
without any asyncio machinery; `run_forever()` is the only async part, and
just wraps it in a poll loop.
"""
import asyncio
import logging

from sqlmodel import Session, select

from app.db import create_alert
from app.db.engine import get_engine
from app.db.models import Camera
from app.services.ws_manager import ws_manager
from app.stream_recorder import stream_recorder_manager

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 30

# ClipRecorder.status ("starting"|"connected"|"reconnecting"|"stopped") ->
# Camera.connectivity_status ("active"|"inactive"|"pending"). A recorder with
# no entry at all (recording disabled, or camera_ip unset) means "disabled",
# which is not monitored here at all — see the recording_enabled filter below.
_STATUS_MAP: dict[str, str] = {
    "connected": "active",
    "starting": "pending",
    "reconnecting": "inactive",
    "stopped": "inactive",
}


def _check_once() -> None:
    with Session(get_engine()) as session:
        cameras = session.exec(
            select(Camera).where(Camera.recording_enabled == True)  # noqa: E712
        ).all()

        for cam in cameras:
            stream_status = stream_recorder_manager.status_for(cam.camera_id)["status"]
            new_status = _STATUS_MAP.get(stream_status, "pending")
            previous = cam.connectivity_status
            if new_status == previous:
                continue

            cam.connectivity_status = new_status
            session.add(cam)
            session.commit()

            if previous == "active" and new_status == "inactive":
                message = f"Camera '{cam.name}' went offline."
                try:
                    create_alert(
                        job_id=None,
                        kpi_name="system",
                        alert_type="camera_offline",
                        frame_idx=None,
                        confidence=1.0,
                        extra={"camera_id": cam.camera_id, "camera_name": cam.name, "message": message},
                        camera_id=cam.camera_id,
                    )
                except Exception:
                    logger.exception("[camera_heartbeat] failed to create offline alert for %s", cam.camera_id)

                ws_manager.broadcast_threadsafe(
                    "camera.offline",
                    {"camera_id": cam.camera_id, "camera_name": cam.name, "connectivity_status": new_status},
                    camera_id=cam.camera_id,
                )
                logger.warning("[camera_heartbeat] %s", message)

            elif previous == "inactive" and new_status == "active":
                # Recovery signal — no Alert row (the plan only calls for one
                # on going offline), just lets connected clients clear any
                # "offline" indicator they're showing for this camera. Not
                # fired for the normal pending->active startup transition.
                ws_manager.broadcast_threadsafe(
                    "camera.online",
                    {"camera_id": cam.camera_id, "camera_name": cam.name, "connectivity_status": new_status},
                    camera_id=cam.camera_id,
                )


async def run_forever() -> None:
    while True:
        try:
            _check_once()
        except Exception:
            logger.exception("[camera_heartbeat] check failed")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
