import cv2
from ultralytics import YOLO

from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ...config import settings

_DEFAULT_CONF           = 0.35
_DEFAULT_MIN_BOX_AREA   = 800
_DEFAULT_MAX_PILLAR_R   = 4.0
_DEFAULT_MIN_PERSON_R   = 0.6


@register_kpi
class PeopleCountKPI(BaseKPI):
    name         = "people_count"
    display_name = "People Count"

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path       = self._get("model_path",       "app/models/ppl-count-yolo26m.pt")
        conf             = self._get("confidence",       _DEFAULT_CONF)
        min_box_area     = self._get("min_box_area",     _DEFAULT_MIN_BOX_AREA)
        max_pillar_ratio = self._get("max_pillar_ratio", _DEFAULT_MAX_PILLAR_R)
        min_person_ratio = self._get("min_person_ratio", _DEFAULT_MIN_PERSON_R)
        frame_stride     = max(1, self._get("frame_stride", 2))

        model = YOLO(model_path)
        cap   = cv2.VideoCapture(video_path)

        unique_ids: set[int] = set()
        alert_events = 0
        frame_idx    = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self._observe(frame, frame_idx, job_id)

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            results = model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                conf=conf, device=device, half=half, verbose=False,
            )
            if not results:
                frame_idx += 1
                continue

            boxes = results[0].boxes
            if boxes is None or boxes.id is None:
                frame_idx += 1
                continue

            track_ids  = boxes.id.int().cpu().tolist()
            cls_ids    = boxes.cls.int().cpu().tolist()
            xyxy_list  = boxes.xyxy.int().cpu().tolist()
            confs      = boxes.conf.cpu().tolist()

            for i in range(len(track_ids)):
                if cls_ids[i] != 0 or confs[i] < conf:
                    continue
                x1, y1, x2, y2 = xyxy_list[i]
                w = x2 - x1; h = y2 - y1
                if w <= 0 or h <= 0:
                    continue
                area = w * h
                asp  = h / (w + 1e-6)
                if area < min_box_area or asp > max_pillar_ratio or asp < min_person_ratio:
                    continue

                tid = track_ids[i]
                if tid not in unique_ids:
                    unique_ids.add(tid)
                    alert_events += 1
                    self._save_alert(
                        "new_person_detected", job_id, frame_idx,
                        confidence=round(confs[i], 3),
                        extra={"track_id": int(tid)},
                        boxes=[(x1, y1, x2, y2, f"ID {tid}", (0, 255, 0))],
                    )

            frame_idx += 1

        cap.release()
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "alert_events":     alert_events,
            "total_foot_traffic": len(unique_ids),
            "total_frames":     frame_idx,
            "device":           device,
        })
