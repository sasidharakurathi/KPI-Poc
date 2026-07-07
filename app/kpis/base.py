
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from .. import db
from ..adaptive_controller import adaptive_controller
from .shared_inference import SharedInference
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
        self.shared_cache = SharedInference()

        # Set by pipeline.run_pipeline() before process_video() runs, when the
        # job belongs to a camera — enables the adaptive frame-gating below.
        # Stays None for camera-less manual uploads (gating becomes a no-op).
        self.camera_id: Optional[str] = None
        self._motion_gate_frame: Optional[np.ndarray] = None

    def _get(self, key: str, default: Any = None) -> Any:
        return self._cfg.get(key, default)

    # ── adaptive frame gating ──────────────────────────────────────────────────

    def _effective_stride(self, base_stride: int) -> int:
        """base_stride widened by this camera's current adaptive skip
        multiplier (see adaptive_controller) — never narrower than configured,
        since that's the finest granularity the KPI was tuned to need."""
        base_stride = max(1, base_stride)
        if not self.camera_id:
            return base_stride
        multiplier = adaptive_controller.get_multiplier(self.camera_id)
        return max(base_stride, round(base_stride * multiplier))

    def _should_process_frame(
        self, frame: np.ndarray, frame_idx: int, base_stride: int, motion_threshold: float = 6.0,
    ) -> bool:
        """Combines two techniques to decide whether to run inference on this
        frame, replacing a plain `frame_idx % frame_stride` check:

        1. Adaptive stride — base_stride is widened under sustained processing
           load for this camera (see _effective_stride), so an overloaded
           camera automatically thins out how many frames it inspects to
           catch back up with the live recording rate.
        2. Motion override — even on a stride-skipped frame, if the scene
           changed enough since the last frame actually processed (something
           entered/left view), it's processed anyway. A busy scene is never
           silently starved just because the stride said to skip it; only
           genuinely static frames are skipped more aggressively under load.

        Uses a 24x24 downsampled grayscale mean-abs-diff — a few orders of
        magnitude cheaper than running inference, so checking every frame
        (even ones ultimately skipped) is negligible next to model cost.
        """
        stride = self._effective_stride(base_stride)
        on_stride = (frame_idx % stride == 0)

        small = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (24, 24), interpolation=cv2.INTER_AREA
        ).astype(np.int16)

        if self._motion_gate_frame is None:
            self._motion_gate_frame = small
            return True   # always process the first frame

        changed = np.abs(small - self._motion_gate_frame).mean() >= motion_threshold
        if on_stride or changed:
            self._motion_gate_frame = small
            return True
        return False

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

        try:
            from ..notifications import notify_alert
            # Find anchor frame and draw detection boxes on it for the email.
            anchor_img = next(
                (img for fidx, img in p.frames if fidx == p.anchor),
                p.frames[0][1] if p.frames else None,
            )
            frame_bytes: Optional[bytes] = None
            if anchor_img is not None:
                labeled = anchor_img.copy()
                if p.boxes:
                    # Draw only rectangles — no text labels — so no confidence
                    # values bleed into the email regardless of what the KPI
                    # put in the label string.
                    for b in p.boxes:
                        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                        color = b[5] if len(b) > 5 else (0, 0, 255)
                        cv2.rectangle(labeled, (x1, y1), (x2, y2), color, 3)
                ok, buf = cv2.imencode(".jpg", labeled, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    frame_bytes = buf.tobytes()
            notify_alert(
                kpi_name=self.name,
                display_name=getattr(self, "display_name", self.name),
                alert_type=p.alert_type,
                job_id=self._job_id,
                alert_id=alert_id,
                confidence=float(p.confidence),
                frame_bytes=frame_bytes,
            )
        except Exception:
            logger.exception("[%s] failed to dispatch email notification", self.name)

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
