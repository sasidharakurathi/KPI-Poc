import cv2
import numpy as np
import supervision as sv

from ... import model_registry
from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ...config import settings

_BATCH_SIZE = 4


@register_kpi
class BoxCounterKPI(BaseKPI):
    name = "object_detection"
    display_name = "Object Detection and Counting"

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path         = self._get("model_path",  "app/models/carton-box-detection.pt")
        conf                = self._get("confidence",  0.75)
        frame_stride        = max(1, self._get("frame_stride", 2))
        min_confirm_frames  = max(1, self._get("min_confirm_frames", 2))
        infer_imgsz         = self._get("infer_imgsz", 640)

        model   = model_registry.get_model(model_path)
        tracker = sv.ByteTrack()
        cap     = cv2.VideoCapture(video_path)

        track_seen:    dict[int, int] = {}
        confirmed_ids: set[int]       = set()
        alert_events = 0
        frame_idx    = 0
        batch: list[np.ndarray] = []

        def _process_result(r) -> None:
            nonlocal alert_events
            sv_dets = sv.Detections.from_ultralytics(r)
            if len(sv_dets) > 0:
                sv_dets = tracker.update_with_detections(sv_dets)

            if len(sv_dets) == 0 or sv_dets.tracker_id is None:
                return

            for i in range(len(sv_dets)):
                tid = int(sv_dets.tracker_id[i])
                if tid in confirmed_ids:
                    continue
                track_seen[tid] = track_seen.get(tid, 0) + 1
                if track_seen[tid] < min_confirm_frames:
                    continue

                confirmed_ids.add(tid)
                alert_events += 1

        def _flush_batch() -> None:
            nonlocal batch
            if not batch:
                return
            results_list = model(batch, conf=conf, imgsz=infer_imgsz, device=device, half=half, verbose=False)
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
            "alert_events":    alert_events,
            "objects_tracked": len(confirmed_ids),
            "total_frames":    frame_idx,
            "device":          device,
        })
