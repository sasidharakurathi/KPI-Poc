"""Decodes a video once and fans frames out to N consumer queues, so multiple KPIs can share one decode instead of each opening its own capture."""
import logging
import queue
import threading
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_QUEUE_MAXSIZE = 32
_SENTINEL = None


class SharedFrameSource:
    def __init__(self, video_path: str, num_consumers: int) -> None:
        self.video_path = video_path
        self._queues: list["queue.Queue"] = [
            queue.Queue(maxsize=_QUEUE_MAXSIZE) for _ in range(num_consumers)
        ]
        self._closed = [threading.Event() for _ in range(num_consumers)]
        self._thread = threading.Thread(
            target=self._decode_loop, daemon=True, name="shared-frame-decoder"
        )
        self._error: Optional[Exception] = None

    def start(self) -> None:
        self._thread.start()

    def close_consumer(self, consumer_idx: int) -> None:
        """Stop feeding and drain this consumer's queue so the decoder can't block on it. Safe to call more than once."""
        self._closed[consumer_idx].set()
        q = self._queues[consumer_idx]
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def _decode_loop(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame.flags.writeable = False   # shared across consumers; guard against in-place mutation
                for i, q in enumerate(self._queues):
                    if self._closed[i].is_set():
                        continue
                    q.put((frame_idx, frame))
                frame_idx += 1
        except Exception as exc:
            self._error = exc
            logger.exception(f"[frame-source] decode failed for {self.video_path}")
        finally:
            cap.release()
            for i, q in enumerate(self._queues):
                if not self._closed[i].is_set():
                    q.put(_SENTINEL)

    def iter_frames(self, consumer_idx: int) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_idx, frame) pairs in order for one consumer. Raises
        if the shared decode loop itself failed."""
        q = self._queues[consumer_idx]
        while True:
            item = q.get()
            if item is _SENTINEL:
                if self._error is not None:
                    raise RuntimeError(
                        f"shared frame decode failed for {self.video_path}"
                    ) from self._error
                return
            yield item
