"""Thins an already-saved video toward a target fps, preserving duration (same trade as stream_recorder.py's live thinning)."""
import os
import shutil
import subprocess
from pathlib import Path

import cv2

from .config import settings


def _cv2_can_decode(path: str) -> bool:
    cap = cv2.VideoCapture(path)
    try:
        ok, _ = cap.read()
        return ok
    finally:
        cap.release()


def _remux_decodable_stream(src_path: str, dst_path: str) -> None:
    """Lossless copy of the first video stream OpenCV can actually decode
    into an mp4. The site's NVR .mkv exports carry an empty h264 track as
    stream 0 with the real footage in a second (mjpeg) stream, and OpenCV
    only ever reads stream 0 - so it gets 0 frames. The mp4 copy also gives
    OpenCV the true frame rate (the .mkv advertises a bogus 60 fps)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            f"OpenCV can't decode '{Path(src_path).name}' and ffmpeg isn't on PATH to repair it - "
            "install ffmpeg or upload an mp4."
        )
    for i in range(8):
        proc = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", src_path, "-map", f"0:v:{i}", "-c", "copy", "-an", dst_path],
            capture_output=True, text=True,
        )
        if "matches no streams" in proc.stderr:
            break
        if proc.returncode == 0 and _cv2_can_decode(dst_path):
            return
    raise RuntimeError(f"no video stream in '{Path(src_path).name}' could be decoded, even after an ffmpeg remux")


def thin_video(src_path: str, dst_path: str) -> None:
    remuxed = None
    if not _cv2_can_decode(src_path):
        remuxed = str(Path(dst_path).with_name(f"{Path(dst_path).stem}_remux.mp4"))
        _remux_decodable_stream(src_path, remuxed)
        src_path = remuxed

    try:
        written = _thin(src_path, dst_path)
    finally:
        if remuxed:
            try:
                os.remove(remuxed)
            except OSError:
                pass

    if written == 0:
        raise RuntimeError(f"no frames could be decoded from '{Path(src_path).name}'")


def _thin(src_path: str, dst_path: str) -> int:
    cap = cv2.VideoCapture(src_path)
    written = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_fps = settings.FRAME_THINNING_TARGET_FPS
        record_fps = target_fps if target_fps < fps else fps
        keep_ratio = (record_fps / fps) if target_fps < fps else 1.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(dst_path, fourcc, record_fps, (fw, fh))
        try:
            keep_acc = 0.0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                keep_acc += keep_ratio
                if keep_acc >= 1.0:
                    writer.write(frame)
                    written += 1
                    keep_acc -= 1.0
        finally:
            writer.release()
    finally:
        cap.release()
    return written
