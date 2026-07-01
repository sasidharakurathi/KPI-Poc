"""Base class for every KPI.

KPIs no longer produce a rendered video. Instead, on each detection a KPI calls
``self._save_alert(...)`` and the base class saves an 8-frame *raw* clip window
(no labels) centred on the trigger frame, plus one row in the database.

How the sliding window works:
  * the KPI loop calls ``self._observe(frame, frame_idx, job_id)`` for EVERY frame
    read (including skipped/strided frames) so the buffer holds consecutive raw frames;
  * ``self._save_alert(...)`` snapshots the buffered "before" frames (incl. the anchor)
    and waits for the next ``ALERT_WINDOW_AFTER`` frames to arrive;
  * once the trailing frames are in, the whole window is written to disk + DB.
  * ``self._finalize()`` (called after the loop) flushes any window still waiting.
"""
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from .. import db
from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class KPIResult:
    kpi_name: str
    display_name: str
    summary: dict[str, Any] = field(default_factory=dict)


_LABEL_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_COLOR = (0, 0, 255)


def _draw_boxes(img: np.ndarray, boxes) -> None:
    """Draw detection boxes/labels for developer-mode labeled frames.
    Each box: (x1, y1, x2, y2, label[, (B,G,R)])."""
    for b in boxes:
        x1, y1, x2, y2, label = int(b[0]), int(b[1]), int(b[2]), int(b[3]), str(b[4])
        color = b[5] if len(b) > 5 else _LABEL_COLOR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(img, label, (x1, max(12, y1 - 6)),
                        _LABEL_FONT, 0.5, color, 2, cv2.LINE_AA)


class _PendingWindow:
    """A detection awaiting its trailing frames before being written out."""
    __slots__ = ("alert_type", "anchor", "confidence", "extra", "frames",
                 "need_after", "boxes")

    def __init__(self, alert_type, anchor, confidence, extra, before_frames,
                 need_after, boxes):
        self.alert_type = alert_type
        self.anchor = anchor
        self.confidence = confidence
        self.extra = extra
        self.frames = list(before_frames)   # [(frame_idx, image), ...] incl. anchor
        self.need_after = need_after
        self.boxes = boxes or []            # anchor detection boxes (developer mode)


class BaseKPI(ABC):
    """
    Subclasses set:
        name         - unique snake_case identifier
        display_name - human-readable name

    Inside process_video(), per frame:
        self._observe(frame, frame_idx, job_id)   # every frame, before processing
        self._save_alert(alert_type, job_id,      # on a detection
                         frame_idx, confidence, extra, boxes)
    After the loop:
        self._finalize()
        return KPIResult(self.name, self.display_name, summary)
    """

    name: str
    display_name: str

    def __init__(self) -> None:
        from ..config_loader import get_kpi_config
        self._cfg: dict[str, Any] = get_kpi_config(self.__class__.__name__)
        self._before: int = max(0, settings.ALERT_WINDOW_BEFORE)
        self._after: int  = max(0, settings.ALERT_WINDOW_AFTER)
        self._buf: deque  = deque(maxlen=self._before + 1)
        self._pending: list[_PendingWindow] = []
        self._job_id: str = ""

    def _get(self, key: str, default: Any = None) -> Any:
        return self._cfg.get(key, default)

    # ── sliding-window frame capture ──────────────────────────────────────────

    def _observe(self, frame: np.ndarray, frame_idx: int, job_id: str = "") -> None:
        """Feed every RAW frame in order (call once per read, before processing)."""
        if job_id:
            self._job_id = job_id

        if self._pending:
            still: list[_PendingWindow] = []
            for p in self._pending:
                p.frames.append((frame_idx, frame.copy()))
                p.need_after -= 1
                if p.need_after <= 0:
                    self._flush(p)
                else:
                    still.append(p)
            self._pending = still

        self._buf.append((frame_idx, frame.copy()))

    def _save_alert(
        self,
        alert_type: str,
        job_id: str,
        frame_idx: int,
        confidence: float = 1.0,
        extra: Optional[dict] = None,
        boxes: Optional[list] = None,
        **_ignored: Any,
    ) -> None:
        """Register a detection. Saves the centred raw-frame window to disk + DB.
        In developer mode, also saves a labeled copy of the anchor frame."""
        if job_id:
            self._job_id = job_id
        before_frames = list(self._buf)
        p = _PendingWindow(alert_type, frame_idx, confidence, extra,
                           before_frames, self._after, boxes)
        if self._after <= 0:
            self._flush(p)
        else:
            self._pending.append(p)

    def _finalize(self) -> None:
        """Flush any windows still awaiting trailing frames (end of video)."""
        for p in self._pending:
            self._flush(p)
        self._pending = []

    def _flush(self, p: _PendingWindow) -> None:
        if not self._job_id or not p.frames:
            return
        try:
            alert_id = db.create_alert(
                job_id=self._job_id,
                kpi_name=self.name,
                alert_type=p.alert_type,
                frame_idx=p.anchor,
                confidence=float(p.confidence),
                extra=p.extra,
            )
        except Exception:
            logger.exception("[%s] failed to persist alert", self.name)
            return

        out_dir = settings.ALERTS_DIR / self._job_id / self.name / f"{alert_id:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        dev_mode = settings.DEV_MODE and bool(p.boxes)

        records: list[tuple[int, int, str, Optional[str]]] = []
        for pos, (fidx, img) in enumerate(p.frames):
            fpath = out_dir / f"{pos:02d}_frame{fidx:06d}.jpg"
            cv2.imwrite(str(fpath), img)

            labeled_path = None
            if dev_mode and fidx == p.anchor:
                labeled_img = img.copy()
                _draw_boxes(labeled_img, p.boxes)
                lpath = out_dir / f"labeled_frame{fidx:06d}.jpg"
                cv2.imwrite(str(lpath), labeled_img)
                labeled_path = str(lpath)

            records.append((pos, fidx, str(fpath), labeled_path))

        try:
            db.add_alert_frames(alert_id, records)
        except Exception:
            logger.exception("[%s] failed to persist alert frames", self.name)

    @abstractmethod
    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        """Run inference over the video. Call _observe() every frame,
        _save_alert() on detections, _finalize() after the loop."""
