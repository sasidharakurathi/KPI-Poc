"""Thins an already-saved video: keeps every other frame and halves the declared fps, preserving duration (same trade as stream_recorder.py's live thinning)."""
import cv2


def thin_video(src_path: str, dst_path: str) -> None:
    cap = cv2.VideoCapture(src_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        record_fps = fps / 2.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(dst_path, fourcc, record_fps, (fw, fh))
        try:
            raw_frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if raw_frame_idx % 2 == 0:
                    writer.write(frame)
                raw_frame_idx += 1
        finally:
            writer.release()
    finally:
        cap.release()
