"""Worker 1 — continuous IP camera capture. One thread per camera rolls RTSP frames into fixed-length clips and hands each off to clip_processor."""
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2

from .config import settings

logger = logging.getLogger(__name__)

# RTSP over TCP: UDP packet loss corrupts the H.264 bitstream mid-GOP; TCP retransmits instead.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

_OPEN_TIMEOUT_MSEC = 10_000
_READ_TIMEOUT_MSEC = 5_000

_MAX_CONSECUTIVE_READ_FAILURES = 6   # ~30s of stalled/failed reads before reconnecting


def build_stream_url(cam) -> Optional[str]:
    """Assembles an rtsp:// URL from a Camera row; None if no IP is configured."""
    if not cam.camera_ip:
        return None

    auth = ""
    if cam.stream_username:
        password = ""
        if cam.stream_password_encrypted:
            from .email_crypto import decrypt_secret, EmailCryptoNotConfigured
            try:
                password = decrypt_secret(cam.stream_password_encrypted)
            except EmailCryptoNotConfigured:
                logger.warning(
                    "[stream:%s] stream password can't be decrypted — connecting without it",
                    cam.camera_id,
                )
        auth = f"{cam.stream_username}:{password}@" if password else f"{cam.stream_username}@"

    path = cam.stream_path or ""
    if path and not path.startswith("/"):
        path = "/" + path
    return f"rtsp://{auth}{cam.camera_ip}:{cam.rtsp_port}{path}"


class ClipRecorder:
    """Owns one camera's capture thread; reconnects on drop, flushing any in-progress clip first."""

    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        stream_url: str,
        on_clip_ready: Callable[[str, str, str, float], None],
    ) -> None:
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.stream_url = stream_url
        self._on_clip_ready = on_clip_ready
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.status = "starting"   # starting | connected | reconnecting | stopped
        self.last_error: Optional[str] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"recorder-{self.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        self.status = "stopped"

    def _run(self) -> None:
        out_dir = settings.RECORDINGS_DIR / self.camera_id
        out_dir.mkdir(parents=True, exist_ok=True)

        while not self._stop.is_set():
            # cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG, [
            #     cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, _OPEN_TIMEOUT_MSEC,
            #     cv2.CAP_PROP_READ_TIMEOUT_MSEC, _READ_TIMEOUT_MSEC,
            # ])
            cap = cv2.VideoCapture(self.stream_url)
            if not cap.isOpened():
                self.status = "reconnecting"
                self.last_error = "failed to open stream"
                cap.release()
                logger.warning(f"[recorder:{self.camera_id}] {self.last_error} — retrying")
                if self._stop.wait(settings.STREAM_RECONNECT_DELAY):
                    break
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 1 or fps > 60:
                fps = 15.0   # many IP cams misreport FPS via RTSP metadata
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

            # Keep every other frame + halve fps (duration unchanged) to cut processing cost.
            thinning_enabled = settings.STREAM_FRAME_THINNING_ENABLED
            record_fps = fps / 2.0 if thinning_enabled else fps

            self.status = "connected"
            self.last_error = None
            logger.info(
                f"[recorder:{self.camera_id}] connected ({fw}x{fh}@{fps:.1f}fps source, "
                + (f"recording thinned to {record_fps:.1f}fps)"
                   if thinning_enabled else "recording at original fps, thinning disabled)")
            )

            writer: Optional[cv2.VideoWriter] = None
            clip_path: Optional[Path] = None
            clip_started = 0.0
            consecutive_failures = 0
            raw_frame_idx = 0

            try:
                while not self._stop.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        consecutive_failures += 1
                        if consecutive_failures >= _MAX_CONSECUTIVE_READ_FAILURES:
                            raise RuntimeError("stream stalled — too many failed reads")
                        time.sleep(0.05)
                        continue
                    consecutive_failures = 0
                    keep_frame = (raw_frame_idx % 2 == 0) if thinning_enabled else True
                    raw_frame_idx += 1

                    now = time.monotonic()
                    if writer is None:
                        clip_started = now
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        clip_path = out_dir / f"{self.camera_id}_{ts}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(clip_path), fourcc, record_fps, (fw, fh))

                    if keep_frame:
                        writer.write(frame)

                    if now - clip_started >= settings.STREAM_CLIP_SECONDS:
                        writer.release()
                        writer = None
                        self._hand_off_clip(str(clip_path), now - clip_started)
                        clip_path = None
            except Exception as exc:
                self.status = "reconnecting"
                self.last_error = str(exc)
                logger.warning(f"[recorder:{self.camera_id}] {exc} — reconnecting")
            finally:
                if writer is not None:
                    writer.release()
                    if clip_path is not None:
                        self._hand_off_clip(str(clip_path), time.monotonic() - clip_started)
                cap.release()

            if not self._stop.is_set():
                self._stop.wait(settings.STREAM_RECONNECT_DELAY)

        self.status = "stopped"
        logger.info(f"[recorder:{self.camera_id}] stopped")

    def _hand_off_clip(self, clip_path: str, duration_sec: float) -> None:
        """Failure here is a queueing problem, not a stream problem -- must not trigger a reconnect."""
        try:
            self._on_clip_ready(self.camera_id, self.camera_name, clip_path, duration_sec)
        except Exception:
            logger.exception(f"[recorder:{self.camera_id}] failed to queue clip {clip_path}")


class StreamRecorderManager:
    """Starts/stops one ClipRecorder per camera; call sync_camera() after any camera create/update/delete."""

    def __init__(self) -> None:
        self._recorders: dict[str, ClipRecorder] = {}
        self._lock = threading.Lock()

    def sync_camera(self, camera_id: str) -> None:
        from . import db
        cam = db.get_camera(camera_id)

        with self._lock:
            existing = self._recorders.get(camera_id)
            url = build_stream_url(cam) if cam else None
            should_run = bool(cam and cam.recording_enabled and url)

            if not should_run:
                if existing:
                    existing.stop()
                    del self._recorders[camera_id]
                return

            if existing and existing.stream_url == url:
                return  # already running with this exact config

            if existing:
                existing.stop()

            from .clip_processor import clip_processor
            recorder = ClipRecorder(
                camera_id=cam.camera_id,
                camera_name=cam.name,
                stream_url=url,
                on_clip_ready=clip_processor.enqueue,
            )
            recorder.start()
            self._recorders[camera_id] = recorder

    def start_all(self) -> None:
        from . import db
        for cam in db.list_cameras():
            self.sync_camera(cam.camera_id)

    def stop_all(self) -> None:
        with self._lock:
            for rec in self._recorders.values():
                rec.stop()
            self._recorders.clear()

    def status_for(self, camera_id: str) -> dict:
        rec = self._recorders.get(camera_id)
        if not rec:
            return {"status": "disabled", "error": None}
        return {"status": rec.status, "error": rec.last_error}


stream_recorder_manager = StreamRecorderManager()
