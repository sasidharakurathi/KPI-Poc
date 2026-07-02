import re
import cv2
import easyocr
from ultralytics import YOLO

from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ...config import settings

_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

_ocr_reader: easyocr.Reader | None = None


def _get_reader(gpu: bool) -> easyocr.Reader:
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
    return _ocr_reader


def _is_valid_plate(text: str, min_chars: int = 4) -> bool:
    return len(re.sub(r'[^A-Z0-9]', '', text.upper())) >= min_chars


def _ocr_plate(plate_img, reader: easyocr.Reader) -> str:
    gray  = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results = reader.readtext(bw, detail=0, paragraph=True, allowlist=_ALLOWLIST)
    return " ".join(results).strip()


@register_kpi
class AnprLprKPI(BaseKPI):
    name         = "ANPR_LPR"
    display_name = "ANPR / License Plate"

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"
        gpu    = device != "cpu"

        model_path  = self._get("model_path",       "app/models/anpr_lpr.pt")
        conf        = self._get("confidence",        0.30)
        aspect_min  = self._get("aspect_ratio_min",  0.5)
        aspect_max  = self._get("aspect_ratio_max",  7.0)
        min_chars   = self._get("min_plate_chars",   4)
        frame_skip  = max(1, self._get("frame_skip", 2))
        infer_imgsz = self._get("infer_imgsz",       640)

        model  = YOLO(model_path)
        reader = _get_reader(gpu)
        cap    = cv2.VideoCapture(video_path)

        plate_cache: dict[int, str] = {}
        seen_plates: set[str]       = set()
        alert_events = 0
        frame_idx    = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self._observe(frame, frame_idx, job_id)

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            results = model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                conf=conf, imgsz=infer_imgsz, device=device, half=half, verbose=False,
            )
            if not results or results[0].boxes is None:
                frame_idx += 1
                continue

            boxes     = results[0].boxes
            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
            xyxy_list = boxes.xyxy.int().cpu().tolist()
            confs     = boxes.conf.cpu().tolist()

            for i, tid in enumerate(track_ids):
                x1, y1, x2, y2 = xyxy_list[i]
                w, h = x2 - x1, y2 - y1
                if h == 0 or not (aspect_min <= w / h <= aspect_max):
                    continue

                if tid in plate_cache:
                    continue  # already alerted for this track

                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue
                plate_text = _ocr_plate(plate_crop, reader)
                if not plate_text or not _is_valid_plate(plate_text, min_chars):
                    continue

                plate_cache[tid] = plate_text
                if plate_text in seen_plates:
                    continue
                seen_plates.add(plate_text)
                alert_events += 1
                self._save_alert(
                    "license_plate_detected", job_id, frame_idx,
                    confidence=confs[i],
                    extra={"plate_text": plate_text, "track_id": int(tid)},
                    boxes=[(x1, y1, x2, y2, f"Plate {tid} {confs[i]:.2f}", (0, 255, 0))],
                )

            frame_idx += 1

        cap.release()
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "alert_events":    alert_events,
            "unique_plates":   len(seen_plates),
            "plates_seen":     sorted(seen_plates),
            "alarm_triggered": alert_events > 0,
            "total_frames":    frame_idx,
        })
