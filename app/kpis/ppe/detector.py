import cv2
import numpy as np
import supervision as sv
from collections import defaultdict
from ultralytics import YOLO

from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ..pose_utils import _DEFAULT_POSE_MODEL_PATH, load_pose_model, run_pose, human_confirmed_in_box
from ...config import settings

PERSON_CLS = "person"
HELMET_CLS = "helmet"
VEST_CLS   = "vest"


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

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path      = self._get("model_path",         "app/models/ppe.pt")
        pose_model_path = self._get("pose_model_path",    _DEFAULT_POSE_MODEL_PATH)
        conf            = self._get("confidence",         0.25)
        margin          = self._get("margin",             0.15)
        overlap_thr     = self._get("overlap_threshold",  0.30)
        alarm_secs      = self._get("alarm_seconds",      2.0)
        alert_hold_secs = self._get("alert_hold_seconds", 4.0)
        frame_stride    = max(1, self._get("frame_stride", 2))
        infer_imgsz     = self._get("infer_imgsz",         640)

        model      = YOLO(model_path)
        pose_model = load_pose_model(pose_model_path)
        tracker    = sv.ByteTrack()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        alarm_frames = int(alarm_secs * fps)

        track_noncompliant: dict[int, int] = defaultdict(int)
        alarmed_ids: set[int] = set()

        compliant = no_helmet = no_vest = no_ppe = 0
        alert_events = 0
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self._observe(frame, frame_idx, job_id)

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            results = model.predict(frame, conf=conf, imgsz=infer_imgsz, device=device, half=half, verbose=False)
            r = results[0]
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

            valid_xyxy, valid_conf, valid_cls = [], [], []
            if len(persons_sv) > 0:
                pose_results = run_pose(pose_model, frame, imgsz=infer_imgsz)
                for i in range(len(persons_sv)):
                    pbox = _box_xyxy(persons_sv.xyxy[i])
                    pconf = float(persons_sv.confidence[i]) if persons_sv.confidence is not None else conf
                    if human_confirmed_in_box(pose_results, int(pbox[0]), int(pbox[1]), int(pbox[2]), int(pbox[3])):
                        valid_xyxy.append(persons_sv.xyxy[i])
                        valid_conf.append(pconf)
                        valid_cls.append(person_cls_id)

            if not valid_xyxy:
                frame_idx += 1
                continue

            valid_sv = sv.Detections(
                xyxy=np.array(valid_xyxy, dtype=np.float32),
                confidence=np.array(valid_conf, dtype=np.float32),
                class_id=np.array(valid_cls, dtype=np.int32),
            )
            tracked = tracker.update_with_detections(valid_sv)

            for i in range(len(tracked)):
                pbox = _box_xyxy(tracked.xyxy[i])
                pconf = float(tracked.confidence[i]) if tracked.confidence is not None else conf
                tracker_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None

                status, color = _raw_status(pbox, helmets, vests, margin, overlap_thr)
                x1, y1, x2, y2 = map(int, pbox)

                if status == "COMPLIANT":   compliant  += 1
                elif status == "NO HELMET": no_helmet  += 1
                elif status == "NO VEST":   no_vest    += 1
                else:                       no_ppe     += 1

                if tracker_id is not None:
                    if status != "COMPLIANT":
                        track_noncompliant[tracker_id] += 1
                    else:
                        track_noncompliant[tracker_id] = max(0, track_noncompliant[tracker_id] - 1)

                    if track_noncompliant[tracker_id] >= alarm_frames and tracker_id not in alarmed_ids:
                        alarmed_ids.add(tracker_id)
                        alert_events += 1
                        self._save_alert(
                            "ppe_non_compliance", job_id, frame_idx,
                            confidence=round(pconf, 3),
                            extra={"tracker_id": tracker_id, "status": status},
                            boxes=[(x1, y1, x2, y2, f"#{tracker_id} {status}", color)],
                        )

            frame_idx += 1

        cap.release()
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "alert_events":           alert_events,
            "compliant_person_frames": compliant,
            "no_helmet_person_frames": no_helmet,
            "no_vest_person_frames":   no_vest,
            "no_ppe_person_frames":    no_ppe,
            "alarm_triggered":         len(alarmed_ids) > 0,
            "total_frames":            frame_idx,
            "device":                  device,
        })
