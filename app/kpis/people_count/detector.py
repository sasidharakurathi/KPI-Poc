import numpy as np

from ... import model_registry
from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ...config import settings

_DEFAULT_CONF           = 0.35
_DEFAULT_MIN_BOX_AREA   = 800
_DEFAULT_MAX_PILLAR_R   = 4.0
_DEFAULT_MIN_PERSON_R   = 0.6

_BATCH_SIZE = 4


@register_kpi
class PeopleCountKPI(BaseKPI):
    name         = "people_count"
    display_name = "People Count"

    def setup(self, video_path: str, job_id: str = "") -> None:
        self._job_id = job_id
        self.device = settings.DEVICE
        self.half   = settings.USE_HALF and self.device != "cpu"

        self.model_path       = self._get("model_path",       "app/models/ppl-count-yolo26m.pt")
        self.conf             = self._get("confidence",       _DEFAULT_CONF)
        self.min_box_area     = self._get("min_box_area",     _DEFAULT_MIN_BOX_AREA)
        self.max_pillar_ratio = self._get("max_pillar_ratio", _DEFAULT_MAX_PILLAR_R)
        self.min_person_ratio = self._get("min_person_ratio", _DEFAULT_MIN_PERSON_R)
        self.frame_stride     = max(1, self._get("frame_stride", 2))
        self.min_confirm_frames = max(1, self._get("min_confirm_frames", 2))
        self.infer_imgsz      = self._get("infer_imgsz", 640)

        self.model = model_registry.get_model(self.model_path)
        model_registry.reset_tracker(self.model)   # shared model instance may have stale tracker state

        self.track_seen: dict[int, int] = {}
        self.unique_ids: set[int] = set()
        self.alert_events = 0
        self._frames_seen = 0
        self.batch: list[np.ndarray] = []

    def _process_result(self, results) -> None:
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
            if cls_ids[i] != 0 or confs[i] < self.conf:
                continue
            x1, y1, x2, y2 = xyxy_list[i]
            w = x2 - x1; h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            area = w * h
            asp  = h / (w + 1e-6)
            if area < self.min_box_area or asp > self.max_pillar_ratio or asp < self.min_person_ratio:
                continue

            tid = track_ids[i]
            if tid in self.unique_ids:
                continue
            self.track_seen[tid] = self.track_seen.get(tid, 0) + 1
            if self.track_seen[tid] < self.min_confirm_frames:
                continue

            self.unique_ids.add(tid)
            self.alert_events += 1

    def _flush_batch(self) -> None:
        if not self.batch:
            return
        results_list = self.model.track(
            self.batch, persist=True, tracker="bytetrack.yaml",
            conf=self.conf, imgsz=self.infer_imgsz, device=self.device, half=self.half, verbose=False,
        )
        if results_list:
            for r in results_list:
                self._process_result(r)
        self.batch = []

    def process_frame(self, frame_idx: int, frame: np.ndarray, job_id: str = "") -> None:
        if frame_idx % self.frame_stride == 0:
            self.batch.append(frame)
            if len(self.batch) >= _BATCH_SIZE:
                self._flush_batch()

        self._frames_seen = frame_idx + 1

    def finalize(self) -> KPIResult:
        self._flush_batch()

        return KPIResult(self.name, self.display_name, {
            "alert_events":     self.alert_events,
            "total_foot_traffic": len(self.unique_ids),
            "total_frames":     self._frames_seen,
            "device":           self.device,
        })
