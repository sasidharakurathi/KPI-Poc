from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float
    color: Optional[tuple] = None


@dataclass
class FrameAnnotation:
    frame_idx: int
    detections: list[Detection] = field(default_factory=list)
    status_lines: list[str] = field(default_factory=list)
    extra: Optional[dict] = None


@dataclass
class KPIResult:
    kpi_name: str
    display_name: str
    color: tuple
    frame_annotations: list[FrameAnnotation]
    summary: dict[str, Any]


class BaseKPI(ABC):
    """
    Base class for every KPI.

    Subclasses set:
        name         - unique snake_case identifier
        display_name - shown in the video overlay
        color        - BGR tuple for bounding boxes

    Concrete helpers available inside process_video():
        self._get(key, default)              - read from config.json
        self._save_alert(frame, alert_type,  - persist frame + DB row
                         job_id, frame_idx,
                         confidence, extra)
    """

    name: str
    display_name: str
    color: tuple  # BGR

    def __init__(self) -> None:
        from ..config_loader import get_kpi_config
        self._cfg: dict[str, Any] = get_kpi_config(self.__class__.__name__)

    def _get(self, key: str, default: Any = None) -> Any:
        """Read a parameter from this KPI's config.json section."""
        return self._cfg.get(key, default)

    def _save_alert(
        self,
        frame: np.ndarray,
        alert_type: str,
        job_id: str,
        frame_idx: int,
        confidence: float = 1.0,
        extra: Optional[dict] = None,
        detections: Optional[list] = None,
    ) -> None:
        """
        Persist an alert event.

        Saves the labeled frame as a JPEG under
            storage/alerts/{job_id}/{kpi_name}_{frame_idx:06d}.jpg
        and writes one row to the alerts SQLite table.

        Call this inside process_video() whenever an alert condition is met.
        """
        from pathlib import Path
        from ..alert_db import insert_alert
        from ..config import settings

        labeled = frame.copy()
        _font = cv2.FONT_HERSHEY_SIMPLEX
        for det in (detections or []):
            color = det.color if det.color is not None else self.color
            cv2.rectangle(labeled, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            label = f"{det.label} {det.confidence:.2f}" if det.confidence < 1.0 else det.label
            (tw, th), _ = cv2.getTextSize(label, _font, 0.5, 1)
            bg_y1 = max(0, det.y1 - th - 8)
            cv2.rectangle(labeled, (det.x1, bg_y1), (det.x1 + tw + 4, det.y1), color, -1)
            cv2.putText(
                labeled, label,
                (det.x1 + 2, det.y1 - 4 if det.y1 > 14 else det.y1 + th + 4),
                _font, 0.5, (255, 255, 255), 1,
            )

        alert_dir = settings.UPLOAD_DIR.parent / "alerts" / job_id
        alert_dir.mkdir(parents=True, exist_ok=True)
        frame_path = alert_dir / f"{self.name}_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), labeled)

        insert_alert(
            job_id=job_id,
            kpi_name=self.name,
            alert_type=alert_type,
            frame_idx=frame_idx,
            confidence=confidence,
            frame_path=str(frame_path),
            extra=extra,
        )

    @abstractmethod
    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        """
        Read the video at video_path, run inference on every frame,
        and return a KPIResult with one FrameAnnotation per frame.

        Call self._save_alert(...) whenever an alert condition is detected.
        """
