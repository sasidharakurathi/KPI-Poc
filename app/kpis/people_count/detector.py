import cv2
import numpy as np

from ... import model_registry
from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ...config import settings

_DEFAULT_CONF           = 0.35
_DEFAULT_MIN_BOX_AREA   = 800
_DEFAULT_MAX_PILLAR_R   = 4.0
_DEFAULT_MIN_PERSON_R   = 0.6

_BATCH_SIZE = 8


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
        min_confirm_frames = max(1, self._get("min_confirm_frames", 2))
        infer_imgsz      = self._get("infer_imgsz", 640)

        model = model_registry.get_model(model_path)
        # Preloaded/shared across jobs — clear leftover ByteTrack state from a
        # previous video before our own persist=True loop starts.
        model_registry.reset_tracker(model)
        cap   = cv2.VideoCapture(video_path)

        track_seen: dict[int, int] = {}
        unique_ids: set[int] = set()
        alert_events = 0
        frame_idx    = 0
        batch: list[np.ndarray] = []

        def _process_result(results) -> None:
            nonlocal alert_events
            if results is None:
                return
            boxes = results.boxes
            if boxes is None or boxes.id is None:
                return

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
                if tid in unique_ids:
                    continue
                track_seen[tid] = track_seen.get(tid, 0) + 1
                if track_seen[tid] < min_confirm_frames:
                    continue

                unique_ids.add(tid)
                alert_events += 1

        def _flush_batch() -> None:
            nonlocal batch
            if not batch:
                return
            results_list = model.track(
                batch, persist=True, tracker="bytetrack.yaml",
                conf=conf, imgsz=infer_imgsz, device=device, half=half, verbose=False,
            )
            if results_list:
                for r in results_list:
                    _process_result(r)
            batch = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                batch.append(frame)
                if len(batch) >= _BATCH_SIZE:
                    _flush_batch()

            frame_idx += 1

        _flush_batch()

        cap.release()

        return KPIResult(self.name, self.display_name, {
            "alert_events":     alert_events,
            "total_foot_traffic": len(unique_ids),
            "total_frames":     frame_idx,
            "device":           device,
        })
