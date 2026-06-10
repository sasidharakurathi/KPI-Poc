import re
import cv2
import easyocr
from ultralytics import YOLO

from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ...config import settings

_DEFAULT_CONF            = 0.3
_DEFAULT_ALERT_HOLD_SECS = 3.0
_DEFAULT_ASPECT_MIN      = 0.5
_DEFAULT_ASPECT_MAX      = 7.0
_DEFAULT_MIN_CHARS       = 4
_DEFAULT_FRAME_SKIP      = 2

_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Module-level singleton — initialised once per process, reused across jobs
_ocr_reader: easyocr.Reader | None = None


def _get_reader(gpu: bool) -> easyocr.Reader:
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
    return _ocr_reader


def _is_valid_plate(text: str, min_chars: int = _DEFAULT_MIN_CHARS) -> bool:
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    return len(cleaned) >= min_chars


def _ocr_plate(plate_img, reader: easyocr.Reader) -> str:
    gray  = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # allowlist restricts decoder to alphanumerics only — ~5x faster on CPU
    results = reader.readtext(bw, detail=0, paragraph=True, allowlist=_ALLOWLIST)
    return " ".join(results).strip()


@register_kpi
class AnprLprKPI(BaseKPI):
    name         = "ANPR_LPR"
    display_name = "ANPR / License Plate"
    color        = (0, 255, 0)

    def process_video(self, video_path: str, job_id: str = None) -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"
        gpu    = device != "cpu"

        model_path      = self._get("model_path",         "app/models/anpr_lpr.pt")
        conf            = self._get("confidence",          _DEFAULT_CONF)
        alert_hold_secs = self._get("alert_hold_seconds",  _DEFAULT_ALERT_HOLD_SECS)
        aspect_min      = self._get("aspect_ratio_min",    _DEFAULT_ASPECT_MIN)
        aspect_max      = self._get("aspect_ratio_max",    _DEFAULT_ASPECT_MAX)
        min_chars       = self._get("min_plate_chars",     _DEFAULT_MIN_CHARS)
        frame_skip      = max(1, self._get("frame_skip",   _DEFAULT_FRAME_SKIP))

        model  = YOLO(model_path)
        reader = _get_reader(gpu)
        cap    = cv2.VideoCapture(video_path)

        frame_annotations: list[FrameAnnotation] = []
        frame_idx = 0

        # track_id → validated plate text (OCR runs once per unique plate)
        plate_cache: dict[int, str] = {}

        # propagated to skipped frames so the overlay stays visible
        last_detections:   list[Detection] = []
        last_status_lines: list[str]       = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ── Skip frame: reuse previous detections, no inference ───────────
            if frame_idx % frame_skip != 0:
                frame_annotations.append(FrameAnnotation(
                    frame_idx=frame_idx,
                    detections=list(last_detections),
                    status_lines=list(last_status_lines),
                ))
                frame_idx += 1
                continue

            # ── ByteTrack: stable ID per plate across frames ──────────────────
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf,
                device=device,
                half=half,
                verbose=False,
            )

            detections:   list[Detection] = []
            status_lines: list[str]       = []
            new_plates:   list[str]       = []

            if results and results[0].boxes is not None:
                boxes     = results[0].boxes
                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
                xyxy_list = boxes.xyxy.int().cpu().tolist()
                confs     = boxes.conf.cpu().tolist()

                for i, tid in enumerate(track_ids):
                    x1, y1, x2, y2 = xyxy_list[i]
                    conf_val        = confs[i]

                    w, h = x2 - x1, y2 - y1
                    if h == 0 or not (aspect_min <= w / h <= aspect_max):
                        continue

                    # ── OCR cache: read each plate only once per video ────────
                    if tid in plate_cache:
                        plate_text = plate_cache[tid]
                    else:
                        plate_crop = frame[y1:y2, x1:x2]
                        if plate_crop.size == 0:
                            continue
                        plate_text = _ocr_plate(plate_crop, reader)
                        if not plate_text or not _is_valid_plate(plate_text, min_chars):
                            continue
                        plate_cache[tid] = plate_text
                        new_plates.append(plate_text)

                        self._save_alert(
                            frame,
                            "license_plate_detected",
                            job_id,
                            frame_idx,
                            confidence=conf_val,
                            extra={"plate_text": plate_text},
                        )

                    detections.append(
                        Detection(x1, y1, x2, y2, f"LP: {plate_text}", conf_val)
                    )

            if detections:
                status_lines.append(f"{len(detections)} plate(s) in frame")
            if new_plates:
                status_lines.append(f"New: {', '.join(new_plates)}")

            last_detections   = detections
            last_status_lines = status_lines

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
            summary={
                "total_frames":  frame_idx,
                "unique_plates": len(plate_cache),
                "plates_seen":   list(plate_cache.values()),
            },
        )
