import cv2
import numpy as np
import supervision as sv
from collections import defaultdict

from ... import model_registry
from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ..pose_utils import _DEFAULT_POSE_MODEL_PATH, load_pose_model, human_confirmed_in_box
from ...config import settings

PERSON_CLS = "person"
HELMET_CLS = "helmet"
VEST_CLS   = "vest"

_BATCH_SIZE = 8


def _box_xyxy(box):
    return [float(x) for x in box]


def _expand_box(box, margin=0.15):
    x1, y1, x2, y2 = box
    w = x2 - x1; h = y2 - y1
    return [x1 - margin*w, y1 - margin*h, x2 + margin*w, y2 + margin*h]


def _box_overlap(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    return (iw*ih) / max(1.0, (b[2]-b[0])*(b[3]-b[1]))


def _raw_status(person_box, helmets, vests, margin, thr):
    expanded = _expand_box(person_box, margin)
    helmet_ok = any(_box_overlap(expanded, h) >= thr for h in helmets)
    vest_ok   = any(_box_overlap(expanded, v) >= thr for v in vests)
    if helmet_ok and vest_ok:   return "COMPLIANT",  (0, 180, 0)
    if helmet_ok:               return "NO VEST",    (0, 140, 255)
    if vest_ok:                 return "NO HELMET",  (0, 140, 255)
    return "NO PPE", (0, 0, 255)


@register_kpi
class PPEKPI(BaseKPI):
    name = "ppe"
    display_name = "PPE Compliance"

    def setup(self, video_path: str, job_id: str = "") -> None:
        self._job_id = job_id
        self.device = settings.DEVICE
        self.half   = settings.USE_HALF and self.device != "cpu"

        self.model_path      = self._get("model_path",         "app/models/ppe.pt")
        self.pose_model_path = self._get("pose_model_path",    _DEFAULT_POSE_MODEL_PATH)
        self.conf            = self._get("confidence",         0.25)
        self.margin          = self._get("margin",             0.15)
        self.overlap_thr     = self._get("overlap_threshold",  0.30)
        alarm_secs           = self._get("alarm_seconds",      2.0)
        self.alert_hold_secs = self._get("alert_hold_seconds", 4.0)
        self.frame_stride    = max(1, self._get("frame_stride", 2))
        self.infer_imgsz     = self._get("infer_imgsz",         640)

        self.model      = model_registry.get_model(self.model_path)
        self.pose_model = load_pose_model(self.pose_model_path)
        self.tracker    = sv.ByteTrack()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        self.alarm_frames = int(alarm_secs * fps)

        self.track_noncompliant: dict[int, int] = defaultdict(int)
        self.alarmed_ids: set[int] = set()

        self.compliant = self.no_helmet = self.no_vest = self.no_ppe = 0
        self.alert_events = 0
        self._frames_seen = 0
        self.batch: list[tuple[int, np.ndarray]] = []

    def _process_one(self, fidx: int, frame: np.ndarray, helmets, vests, persons_sv, person_cls_id, pose_results) -> None:
        valid_xyxy, valid_conf, valid_cls = [], [], []
        if len(persons_sv) > 0 and pose_results is not None:
            for i in range(len(persons_sv)):
                pbox = _box_xyxy(persons_sv.xyxy[i])
                pconf = float(persons_sv.confidence[i]) if persons_sv.confidence is not None else self.conf
                if human_confirmed_in_box(pose_results, int(pbox[0]), int(pbox[1]), int(pbox[2]), int(pbox[3])):
                    valid_xyxy.append(persons_sv.xyxy[i])
                    valid_conf.append(pconf)
                    valid_cls.append(person_cls_id)

        if not valid_xyxy:
            return

        valid_sv = sv.Detections(
            xyxy=np.array(valid_xyxy, dtype=np.float32),
            confidence=np.array(valid_conf, dtype=np.float32),
            class_id=np.array(valid_cls, dtype=np.int32),
        )
        tracked = self.tracker.update_with_detections(valid_sv)

        for i in range(len(tracked)):
            pbox = _box_xyxy(tracked.xyxy[i])
            pconf = float(tracked.confidence[i]) if tracked.confidence is not None else self.conf
            tracker_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None

            status, color = _raw_status(pbox, helmets, vests, self.margin, self.overlap_thr)
            x1, y1, x2, y2 = map(int, pbox)

            if status == "COMPLIANT":   self.compliant  += 1
            elif status == "NO HELMET": self.no_helmet  += 1
            elif status == "NO VEST":   self.no_vest    += 1
            else:                       self.no_ppe     += 1

            if tracker_id is not None:
                if status != "COMPLIANT":
                    self.track_noncompliant[tracker_id] += 1
                else:
                    self.track_noncompliant[tracker_id] = max(0, self.track_noncompliant[tracker_id] - 1)

                if self.track_noncompliant[tracker_id] >= self.alarm_frames and tracker_id not in self.alarmed_ids:
                    self.alarmed_ids.add(tracker_id)
                    self.alert_events += 1
                    self._save_alert(
                        "ppe_non_compliance", self._job_id, fidx,
                        confidence=round(pconf, 3),
                        extra={"tracker_id": tracker_id, "status": status},
                        boxes=[(x1, y1, x2, y2, f"#{tracker_id} {status}", color)],
                    )

    def _flush_batch(self) -> None:
        if not self.batch:
            return
        frames = [f for _, f in self.batch]
        results_list = self.model.predict(frames, conf=self.conf, imgsz=self.infer_imgsz, device=self.device, half=self.half, verbose=False)

        stage1 = []   # (fidx, frame, helmets, vests, persons_sv, person_cls_id)
        pose_batch_frames = []
        pose_batch_slots  = []   # index into stage1 needing a pose result
        for (fidx, frame), r in zip(self.batch, results_list):
            sv_dets = sv.Detections.from_ultralytics(r)
            names   = r.names

            person_cls_id = next((k for k, v in names.items() if v == PERSON_CLS), None)
            helmet_cls_id = next((k for k, v in names.items() if v == HELMET_CLS), None)
            vest_cls_id   = next((k for k, v in names.items() if v == VEST_CLS), None)

            helmets = []
            vests   = []
            if len(sv_dets) > 0:
                if helmet_cls_id is not None:
                    helmets = [_box_xyxy(b) for b in sv_dets[sv_dets.class_id == helmet_cls_id].xyxy]
                if vest_cls_id is not None:
                    vests   = [_box_xyxy(b) for b in sv_dets[sv_dets.class_id == vest_cls_id].xyxy]

            persons_sv = sv_dets[sv_dets.class_id == person_cls_id] \
                if person_cls_id is not None and len(sv_dets) > 0 else sv.Detections.empty()

            slot = len(stage1)
            stage1.append((fidx, frame, helmets, vests, persons_sv, person_cls_id))
            if len(persons_sv) > 0:
                pose_batch_frames.append(frame)
                pose_batch_slots.append(slot)

        pose_by_slot: dict[int, list] = {}
        if pose_batch_frames:
            pose_results_list = self.pose_model.predict(
                pose_batch_frames, imgsz=self.infer_imgsz, device=self.device, half=self.half, verbose=False,
            )
            for slot, pres in zip(pose_batch_slots, pose_results_list):
                pose_by_slot[slot] = [pres]   # human_confirmed_in_box expects a list of Results

        for slot, (fidx, frame, helmets, vests, persons_sv, person_cls_id) in enumerate(stage1):
            self._process_one(fidx, frame, helmets, vests, persons_sv, person_cls_id, pose_by_slot.get(slot))

        self.batch = []

    def process_frame(self, frame_idx: int, frame: np.ndarray, job_id: str = "") -> None:
        self._observe(frame, frame_idx, self._job_id)

        if frame_idx % self.frame_stride == 0:
            self.batch.append((frame_idx, frame))
            if len(self.batch) >= _BATCH_SIZE:
                self._flush_batch()

        self._frames_seen = frame_idx + 1

    def finalize(self) -> KPIResult:
        self._flush_batch()
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "alert_events":           self.alert_events,
            "compliant_person_frames": self.compliant,
            "no_helmet_person_frames": self.no_helmet,
            "no_vest_person_frames":   self.no_vest,
            "no_ppe_person_frames":    self.no_ppe,
            "alarm_triggered":         len(self.alarmed_ids) > 0,
            "total_frames":            self._frames_seen,
            "device":                  self.device,
        })
