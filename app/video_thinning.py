"""Thins an already-saved video toward a target fps, preserving duration (same trade as stream_recorder.py's live thinning)."""
import cv2

from .config import settings


def thin_video(src_path: str, dst_path: str) -> None:
    cap = cv2.VideoCapture(src_path)
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
                    keep_acc -= 1.0
        finally:
            writer.release()
    finally:
        cap.release()
