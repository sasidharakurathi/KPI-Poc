import re
import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO

from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ...config import settings

# Default values — used when the key is absent from config.json
_DEFAULT_CONF            = 0.3
_DEFAULT_ALERT_HOLD_SECS = 3.0
_DEFAULT_ASPECT_MIN      = 0.5
_DEFAULT_ASPECT_MAX      = 7.0
_DEFAULT_MIN_CHARS       = 4


def _is_valid_plate(text: str, min_chars: int = _DEFAULT_MIN_CHARS) -> bool:
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    return len(cleaned) >= min_chars


def _extract_plate_text(plate_img, ocr: PaddleOCR) -> str:
    gray   = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, bw  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_bgr = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    result = ocr.predict(bw_bgr)
    text   = ""
    if result:
        for res in result:
            for item in res.get('rec_texts', []):
                text += item + " "
    return text.strip()


@register_kpi
class AnprLprKPI(BaseKPI):
    name         = "ANPR_KPI"
    display_name = "ANPR / License Plate"
    color        = (0, 255, 0)          # BGR green

    def process_video(self, video_path: str, job_id: str = None) -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        # Read every parameter from config.json (second arg is the fallback)
        model_path      = self._get("model_path",         "app/kpis/ANPR-LPR/anpr_lpr.pt")
        conf            = self._get("confidence",          _DEFAULT_CONF)
        alert_hold_secs = self._get("alert_hold_seconds",  _DEFAULT_ALERT_HOLD_SECS)
        aspect_min      = self._get("aspect_ratio_min",    _DEFAULT_ASPECT_MIN)
        aspect_max      = self._get("aspect_ratio_max",    _DEFAULT_ASPECT_MAX)
        min_chars       = self._get("min_plate_chars",     _DEFAULT_MIN_CHARS)

        model = YOLO(model_path)
        ocr   = PaddleOCR(use_angle_cls=True, lang='en')
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25

        frame_annotations = []
        frame_idx         = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                frame, conf=conf, device=device, half=half, verbose=False
            )

            detections   = []
            status_lines = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf_val        = float(box.conf[0])

                    # Aspect ratio filter
                    w = x2 - x1
                    h = y2 - y1
                    if h == 0:
                        continue
                    aspect_ratio = w / h
                    if aspect_ratio < aspect_min or aspect_ratio > aspect_max:
                        continue

                    # Crop plate
                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop.size == 0:
                        continue

                    # OCR
                    plate_text = _extract_plate_text(plate_crop, ocr)

                    # Validity filter
                    if not plate_text or not _is_valid_plate(plate_text, min_chars):
                        continue

                    label = f"License Plate: {plate_text}"
                    detections.append(Detection(x1, y1, x2, y2, label, conf_val))

                    # Save alert with the license plate text as metadata
                    self._save_alert(
                        frame,
                        "license_plate_detected",
                        job_id,
                        frame_idx,
                        confidence=conf_val,
                        extra={"plate_text": plate_text}
                    )

            if detections:
                status_lines.append(f"{len(detections)} plate(s) detected")

            frame_annotations.append(FrameAnnotation(
                frame_idx=frame_idx,
                detections=detections,
                status_lines=status_lines,
            ))
            frame_idx += 1

        cap.release()

        return KPIResult(
            kpi_name=self.name,
            display_name=self.display_name,
            color=self.color,
            frame_annotations=frame_annotations,
            summary={"total_frames": frame_idx},
        )