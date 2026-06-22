import cv2
import numpy as np
import supervision as sv
from collections import defaultdict
from ultralytics import YOLO

from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ..pose_utils import _DEFAULT_POSE_MODEL_PATH, load_pose_model, run_pose, human_keypoints_in_box
from ...config import settings


_DEFAULT_MODEL_PATH        = "app/models/ppe.pt"
_DEFAULT_CONF              = 0.25
_DEFAULT_MARGIN            = 0.15
_DEFAULT_OVERLAP_THR       = 0.30
_DEFAULT_ALARM_SECS        = 2.0
_DEFAULT_ALERT_HOLD_SECS   = 4.0


PERSON_CLS = "person"
HELMET_CLS = "helmet"
VEST_CLS   = "vest"


def box_xyxy(box):
    return [float(x) for x in box]


def expand_box(box, margin=0.15):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return [x1 - margin * w, y1 - margin * h, x2 + margin * w, y2 + margin * h]


def box_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / area_b


def raw_status(person_box, helmets, vests, margin=0.15, thr=0.30):
    expanded = expand_box(person_box, margin)
    helmet_ok = any(box_overlap(expanded, h) >= thr for h in helmets)
    vest_ok   = any(box_overlap(expanded, v) >= thr for v in vests)

    if helmet_ok and vest_ok:
        return "COMPLIANT", (0, 180, 0), expanded
    if helmet_ok and not vest_ok:
        return "NO VEST", (0, 140, 255), expanded
    if vest_ok and not helmet_ok:
        return "NO HELMET", (0, 140, 255), expanded
    return "NO PPE", (0, 0, 255), expanded


@register_kpi
class PPEKPI(BaseKPI):
    name = "ppe"
    display_name = "PPE Compliance"
    color = (0, 255, 0)

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path      = self._get("model_path",          _DEFAULT_MODEL_PATH)
        conf            = self._get("confidence",          _DEFAULT_CONF)
        margin          = self._get("margin",              _DEFAULT_MARGIN)
        overlap_thr     = self._get("overlap_threshold",   _DEFAULT_OVERLAP_THR)
        alarm_secs      = self._get("alarm_seconds",       _DEFAULT_ALARM_SECS)
        alert_hold_secs = self._get("alert_hold_seconds",  _DEFAULT_ALERT_HOLD_SECS)

        pose_model_path = self._get("pose_model_path", _DEFAULT_POSE_MODEL_PATH)

        model      = YOLO(model_path)
        pose_model = load_pose_model(pose_model_path)

        tracker = sv.ByteTrack()

        video_info   = sv.VideoInfo.from_video_path(video_path)
        fps          = video_info.fps or 25
        alarm_frames = int(alarm_secs * fps)
        hold_frames  = int(alert_hold_secs * fps)

        cap = cv2.VideoCapture(video_path)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        from ...kpis.base import get_dynamic_scale
        scale = get_dynamic_scale(w, h) if w and h else 1.0
        
        box_annotator = sv.BoxAnnotator(thickness=max(1, int(round(2 * scale))))
        label_annotator = sv.LabelAnnotator(
            text_scale=0.5 * scale,
            text_thickness=max(1, int(round(1 * scale)))
        )

        frame_annotations: list[FrameAnnotation] = []

        compliant_count = 0
        no_helmet_count = 0
        no_vest_count   = 0
        no_ppe_count    = 0

        track_noncompliant_counts: dict[int, int] = defaultdict(int)
        alarmed_ids: set[int] = set()
        alarm_hold_until: dict[int, int] = {}

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                source=frame,
                device=device,
                half=half,
                conf=conf,
                verbose=False,
            )

            result = results[0]
            sv_detections = sv.Detections.from_ultralytics(result)
            names = result.names

            detections: list[Detection] = []
            labels: list[str] = []

            person_cls_id = next((k for k, v in names.items() if v == PERSON_CLS), None)
            helmet_cls_id = next((k for k, v in names.items() if v == HELMET_CLS), None)
            vest_cls_id   = next((k for k, v in names.items() if v == VEST_CLS), None)

            if person_cls_id is not None and len(sv_detections) > 0:
                persons_sv = sv_detections[sv_detections.class_id == person_cls_id]
            else:
                persons_sv = sv.Detections.empty()

            if helmet_cls_id is not None and len(sv_detections) > 0:
                helmets_sv = sv_detections[sv_detections.class_id == helmet_cls_id]
            else:
                helmets_sv = sv.Detections.empty()

            if vest_cls_id is not None and len(sv_detections) > 0:
                vests_sv = sv_detections[sv_detections.class_id == vest_cls_id]
            else:
                vests_sv = sv.Detections.empty()

            helmets = [box_xyxy(b) for b in helmets_sv.xyxy] if len(helmets_sv) > 0 else []
            vests   = [box_xyxy(b) for b in vests_sv.xyxy] if len(vests_sv) > 0 else []

            valid_person_xyxy = []
            valid_person_conf = []
            valid_person_cls  = []

            if len(persons_sv) > 0:
                pose_results = run_pose(pose_model, frame)

                for i in range(len(persons_sv)):
                    pbox = box_xyxy(persons_sv.xyxy[i])
                    pconf = float(persons_sv.confidence[i]) if persons_sv.confidence is not None else conf

                    if human_keypoints_in_box(
                        pose_results,
                        int(pbox[0]), int(pbox[1]), int(pbox[2]), int(pbox[3])
                    ):
                        valid_person_xyxy.append(persons_sv.xyxy[i])
                        valid_person_conf.append(pconf)
                        valid_person_cls.append(person_cls_id)

            if valid_person_xyxy:
                valid_persons_sv = sv.Detections(
                    xyxy=np.array(valid_person_xyxy, dtype=np.float32),
                    confidence=np.array(valid_person_conf, dtype=np.float32),
                    class_id=np.array(valid_person_cls, dtype=np.int32),
                )
                tracked_persons_sv = tracker.update_with_detections(valid_persons_sv)
            else:
                tracked_persons_sv = sv.Detections.empty()

            annotated_detections_xyxy = []
            annotated_detections_conf = []
            annotated_detections_class = []

            if len(tracked_persons_sv) > 0:
                for i in range(len(tracked_persons_sv)):
                    pbox = box_xyxy(tracked_persons_sv.xyxy[i])
                    pconf = float(tracked_persons_sv.confidence[i]) if tracked_persons_sv.confidence is not None else conf
                    tracker_id = None

                    if tracked_persons_sv.tracker_id is not None and len(tracked_persons_sv.tracker_id) > i:
                        tracker_id = int(tracked_persons_sv.tracker_id[i])

                    status, color, expanded = raw_status(
                        pbox,
                        helmets,
                        vests,
                        margin=margin,
                        thr=overlap_thr,
                    )

                    x1, y1, x2, y2 = map(int, pbox)

                    detections.append(
                        Detection(
                            x1, y1, x2, y2,
                            status,
                            float(pconf),
                            color=color,
                        )
                    )

                    if status == "COMPLIANT":
                        compliant_count += 1
                    elif status == "NO HELMET":
                        no_helmet_count += 1
                    elif status == "NO VEST":
                        no_vest_count += 1
                    elif status == "NO PPE":
                        no_ppe_count += 1

                    if tracker_id is not None and status in ("NO HELMET", "NO VEST", "NO PPE"):
                        track_noncompliant_counts[tracker_id] += 1

                        visible_secs = track_noncompliant_counts[tracker_id] / fps
                        is_alarm = track_noncompliant_counts[tracker_id] >= alarm_frames

                        if is_alarm and tracker_id not in alarmed_ids:
                            alarmed_ids.add(tracker_id)
                            alarm_hold_until[tracker_id] = frame_idx + hold_frames

                            if job_id:
                                self._save_alert(
                                    frame,
                                    "ppe_non_compliance",
                                    job_id,
                                    frame_idx,
                                    extra={
                                        "tracker_id": int(tracker_id),
                                        "visible_secs": float(visible_secs),
                                        "status": status,
                                    },
                                )

                        if tracker_id in alarmed_ids:
                            alarm_hold_until[tracker_id] = frame_idx + hold_frames

                        label = f"#{tracker_id} {status} {visible_secs:.1f}s"
                        if is_alarm:
                            label += " ALARM"
                    else:
                        label = status

                    labels.append(label)
                    annotated_detections_xyxy.append([x1, y1, x2, y2])
                    annotated_detections_conf.append(float(pconf))
                    annotated_detections_class.append(0)

            ppe_alarm_active = any(
                frame_idx <= alarm_hold_until.get(tid, -1)
                for tid in alarmed_ids
            )

            non_compliant_this_frame = any(
                d.label in ("NO HELMET", "NO VEST", "NO PPE")
                for d in detections
            )

            status_lines: list[str] = []
            if ppe_alarm_active:
                status_lines.append("!! PPE NON-COMPLIANCE ALARM")
            elif non_compliant_this_frame:
                status_lines.append("PPE NON-COMPLIANCE DETECTED")

            if annotated_detections_xyxy:
                annotated_sv = sv.Detections(
                    xyxy=np.array(annotated_detections_xyxy, dtype=np.float32),
                    confidence=np.array(annotated_detections_conf, dtype=np.float32),
                    class_id=np.array(annotated_detections_class, dtype=np.int32),
                )
                annotated_frame = box_annotator.annotate(
                    scene=frame.copy(),
                    detections=annotated_sv,
                )
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame,
                    detections=annotated_sv,
                    labels=labels,
                )
            else:
                annotated_frame = frame.copy()

            if ppe_alarm_active:
                cv2.putText(
                    annotated_frame,
                    "!! PPE NON-COMPLIANCE ALARM",
                    (max(10, int(30 * scale)), max(20, int(50 * scale))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0 * scale,
                    (0, 0, 255),
                    max(1, int(round(2 * scale))),
                    cv2.LINE_AA,
                )

            frame_annotations.append(
                FrameAnnotation(
                    frame_idx=frame_idx,
                    detections=detections,
                    status_lines=status_lines,
                )
            )
            frame_idx += 1

        cap.release()

        return KPIResult(
            kpi_name=self.name,
            display_name=self.display_name,
            color=self.color,
            frame_annotations=frame_annotations,
            summary={
                "compliant_person_frames": compliant_count,
                "no_helmet_person_frames": no_helmet_count,
                "no_vest_person_frames": no_vest_count,
                "no_ppe_person_frames": no_ppe_count,
                "alarm_triggered": len(alarmed_ids) > 0,
                "total_frames": frame_idx,
                "device": device,
            },
        )