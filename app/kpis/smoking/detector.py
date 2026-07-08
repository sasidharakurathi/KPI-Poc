"""Smoking KPI -- ByteTrack persons, then a batched cigarette model on upper-body crops; a per-track hit counter (incremented on detection, decayed on miss) fires one alert at consecutive_frames."""
import cv2
import numpy as np
import supervision as sv
from collections import defaultdict

from ... import model_registry
from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ...config import settings

_BATCH_SIZE = 8


@register_kpi
class SmokingKPI(BaseKPI):
    name = "smoking"
    display_name = "Smoking"

    def setup(self, video_path: str, job_id: str = "") -> None:
        self._job_id = job_id
        self.device = settings.DEVICE
        self.half   = settings.USE_HALF and self.device != "cpu"

        self.person_model_path = self._get("person_model_path",   "app/models/yolo26m.pt")
        self.cig_model_path    = self._get("cigarette_model_path","app/models/cigarette.pt")
        self.person_conf       = self._get("person_confidence",   0.40)
        self.cig_conf          = self._get("cigarette_confidence",0.45)
        self.consec_frames     = self._get("consecutive_frames",  8)
        self.max_limit         = self._get("max_counter_limit",   15)
        self.upper_frac        = self._get("upper_body_fraction", 0.60)
        self.cig_imgsz         = self._get("cigarette_imgsz",     320)
        self.person_imgsz      = self._get("person_imgsz",        640)
        self.frame_stride      = max(1, self._get("frame_stride", 3))

        self.cig_model = model_registry.get_model(self.cig_model_path)
        self.tracker   = sv.ByteTrack()

        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self.track_history: dict[int, int] = defaultdict(int)
        self.alarmed_ids:   set[int]       = set()
        self.alert_events = 0
        self._frames_seen = 0
        self.batch: list[tuple[int, np.ndarray]] = []

    def _process_one(self, fidx: int, frame: np.ndarray, raw_boxes: list) -> None:
        person_rows = [
            (x1, y1, x2, y2, conf) for x1, y1, x2, y2, cls_id, conf in raw_boxes
            if cls_id == 0 and conf >= self.person_conf
        ]
        if person_rows:
            sv_dets = sv.Detections(
                xyxy=np.array([r[:4] for r in person_rows], dtype=np.float32),
                confidence=np.array([r[4] for r in person_rows], dtype=np.float32),
                class_id=np.zeros(len(person_rows), dtype=int),
            )
            sv_dets = self.tracker.update_with_detections(sv_dets)
        else:
            sv_dets = sv.Detections.empty()

        if len(sv_dets) == 0 or sv_dets.tracker_id is None:
            return

        crops:        list[np.ndarray] = []
        crop_to_idx:  list[int]        = []

        persons = list(zip(sv_dets.tracker_id, sv_dets.xyxy))
        for pidx, (tid, bbox) in enumerate(persons):
            x1, y1, x2, y2 = map(int, bbox)
            roi_h = int((y2 - y1) * self.upper_frac)
            cy1 = max(0, y1); cy2 = min(self.fh, y1 + roi_h)
            cx1 = max(0, x1); cx2 = min(self.fw, x2)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                crops.append(crop)
                crop_to_idx.append(pidx)

        cig_hit:  dict[int, bool]  = {}
        cig_conf_val: dict[int, float] = {}
        if crops:
            cig_res = self.cig_model(
                crops, conf=self.cig_conf, imgsz=self.cig_imgsz,
                device=self.device, half=self.half, verbose=False,
            )
            for j, cr in enumerate(cig_res):
                pidx = crop_to_idx[j]
                cig_hit[pidx] = len(cr.boxes) > 0
                if len(cr.boxes) > 0:
                    cig_conf_val[pidx] = float(cr.boxes.conf.max())

        for pidx, (tid, bbox) in enumerate(persons):
            if tid is None:
                continue
            tid = int(tid)
            hit = cig_hit.get(pidx, False)
            if hit:
                self.track_history[tid] = min(self.max_limit, self.track_history[tid] + 1)
            else:
                self.track_history[tid] = max(0, self.track_history[tid] - 1)

            if self.track_history[tid] >= self.consec_frames and tid not in self.alarmed_ids:
                self.alarmed_ids.add(tid)
                self.alert_events += 1
                x1, y1, x2, y2 = map(int, bbox)
                self._save_alert(
                    "smoking_alarm", self._job_id, fidx,
                    confidence=round(cig_conf_val.get(pidx, self.cig_conf), 3),
                    extra={"tracker_id": tid, "counter": self.track_history[tid]},
                    boxes=[(x1, y1, x2, y2, f"#{tid} SMOKING", (0, 255, 255))],
                )

    def _flush_batch(self) -> None:
        if not self.batch:
            return
        boxes_by_frame = self.shared_cache.predict_boxes_batch(
            self.person_model_path, self.batch, self.person_imgsz, self.device, self.half
        )
        for fidx, frame in self.batch:
            self._process_one(fidx, frame, boxes_by_frame.get(fidx, []))
        self.batch = []

    def process_frame(self, frame_idx: int, frame: np.ndarray, job_id: str = "") -> None:
        self._observe(frame, frame_idx, self._job_id)

        if frame_idx % self.frame_stride == 0:
            self.batch.append((frame_idx, frame))
            if len(self.batch) >= _BATCH_SIZE:
                self._flush_batch()

        self._frames_seen = frame_idx + 1

    def finalize(self) -> KPIResult:
        self._flush_batch()   # any leftover partial batch at end of video
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "alert_events":        self.alert_events,
            "unique_smokers_found": len(self.alarmed_ids),
            "alarm_triggered":     self.alert_events > 0,
            "total_frames":        self._frames_seen,
            "device":              self.device,
        })
