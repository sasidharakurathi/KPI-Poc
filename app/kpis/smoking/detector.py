"""
Smoking KPI — 2-stage pipeline.

Stage 1: YOLO person detector + ByteTrack → tracked persons
Stage 2: Batched cigarette model on upper-body crops → per-person hit counter

A person is flagged as smoking when their hit counter reaches `consecutive_frames`.
Counter is incremented on detection, decremented on miss (clamped to [0, max_counter_limit]).
One alert fires per track (on first threshold crossing).
"""
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

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        person_model_path = self._get("person_model_path",   "app/models/yolo26m.pt")
        cig_model_path    = self._get("cigarette_model_path","app/models/cigarette.pt")
        person_conf       = self._get("person_confidence",   0.40)
        cig_conf          = self._get("cigarette_confidence",0.45)
        consec_frames     = self._get("consecutive_frames",  8)
        max_limit         = self._get("max_counter_limit",   15)
        upper_frac        = self._get("upper_body_fraction", 0.60)
        cig_imgsz         = self._get("cigarette_imgsz",     320)
        person_imgsz      = self._get("person_imgsz",        640)
        frame_stride      = max(1, self._get("frame_stride", 3))

        cig_model    = model_registry.get_model(cig_model_path)
        tracker      = sv.ByteTrack()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        track_history: dict[int, int] = defaultdict(int)
        alarmed_ids:   set[int]       = set()
        alert_events = 0
        frame_idx    = 0
        batch: list[tuple[int, np.ndarray]] = []

        def _process_one(fidx: int, frame: np.ndarray, raw_boxes: list) -> None:
            nonlocal alert_events

            # ── Stage 1: track persons from this frame's (batched) raw boxes
            person_rows = [
                (x1, y1, x2, y2, conf) for x1, y1, x2, y2, cls_id, conf in raw_boxes
                if cls_id == 0 and conf >= person_conf
            ]
            if person_rows:
                sv_dets = sv.Detections(
                    xyxy=np.array([r[:4] for r in person_rows], dtype=np.float32),
                    confidence=np.array([r[4] for r in person_rows], dtype=np.float32),
                    class_id=np.zeros(len(person_rows), dtype=int),
                )
                sv_dets = tracker.update_with_detections(sv_dets)
            else:
                sv_dets = sv.Detections.empty()

            if len(sv_dets) == 0 or sv_dets.tracker_id is None:
                return

            # ── Stage 2: collect upper-body crops (batched per frame) ─────────
            crops:        list[np.ndarray] = []
            crop_to_idx:  list[int]        = []

            persons = list(zip(sv_dets.tracker_id, sv_dets.xyxy))
            for pidx, (tid, bbox) in enumerate(persons):
                x1, y1, x2, y2 = map(int, bbox)
                roi_h = int((y2 - y1) * upper_frac)
                cy1 = max(0, y1); cy2 = min(fh, y1 + roi_h)
                cx1 = max(0, x1); cx2 = min(fw, x2)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    crops.append(crop)
                    crop_to_idx.append(pidx)

            # ── One batched GPU call ───────────────────────────────────────────
            cig_hit:  dict[int, bool]  = {}
            cig_conf_val: dict[int, float] = {}
            if crops:
                cig_res = cig_model(
                    crops, conf=cig_conf, imgsz=cig_imgsz,
                    device=device, half=half, verbose=False,
                )
                for j, cr in enumerate(cig_res):
                    pidx = crop_to_idx[j]
                    cig_hit[pidx] = len(cr.boxes) > 0
                    if len(cr.boxes) > 0:
                        cig_conf_val[pidx] = float(cr.boxes.conf.max())

            # ── Update counters & fire alerts ──────────────────────────────────
            for pidx, (tid, bbox) in enumerate(persons):
                if tid is None:
                    continue
                tid = int(tid)
                hit = cig_hit.get(pidx, False)
                if hit:
                    track_history[tid] = min(max_limit, track_history[tid] + 1)
                else:
                    track_history[tid] = max(0, track_history[tid] - 1)

                if track_history[tid] >= consec_frames and tid not in alarmed_ids:
                    alarmed_ids.add(tid)
                    alert_events += 1
                    x1, y1, x2, y2 = map(int, bbox)
                    self._save_alert(
                        "smoking_alarm", job_id, fidx,
                        confidence=round(cig_conf_val.get(pidx, cig_conf), 3),
                        extra={"tracker_id": tid, "counter": track_history[tid]},
                        boxes=[(x1, y1, x2, y2, f"#{tid} SMOKING", (0, 255, 255))],
                    )

        def _flush_batch() -> None:
            nonlocal batch
            if not batch:
                return
            boxes_by_frame = self.shared_cache.predict_boxes_batch(
                person_model_path, batch, person_imgsz, device, half
            )
            for fidx, frame in batch:
                _process_one(fidx, frame, boxes_by_frame.get(fidx, []))
            batch = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self._observe(frame, frame_idx, job_id)

            if frame_idx % frame_stride == 0:
                batch.append((frame_idx, frame))
                if len(batch) >= _BATCH_SIZE:
                    _flush_batch()

            frame_idx += 1

        _flush_batch()   # any leftover partial batch at end of video

        cap.release()
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "alert_events":        alert_events,
            "unique_smokers_found": len(alarmed_ids),
            "alarm_triggered":     alert_events > 0,
            "total_frames":        frame_idx,
            "device":              device,
        })
