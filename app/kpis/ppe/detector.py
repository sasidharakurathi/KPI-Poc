import cv2
from ultralytics import YOLO

from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ..pose_utils import _DEFAULT_POSE_MODEL_PATH, load_pose_model, run_pose, human_keypoints_in_box
from ...config import settings


_DEFAULT_MODEL_PATH        = "app/models/ppe.pt"
_DEFAULT_CONF              = 0.25
_DEFAULT_MARGIN            = 0.15
_DEFAULT_OVERLAP_THR       = 0.30
_DEFAULT_ALARM_THRESHOLD   = 10
_DEFAULT_ALERT_HOLD_SECS   = 3.0


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
    """
    Returns:
        status_str: "COMPLIANT" / "NO VEST" / "NO HELMET" / "NO PPE"
        color:      BGR tuple used for drawing this person
        expanded:   expanded person box used for overlap checks
    """
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

        model_path      = self._get("model_path",            _DEFAULT_MODEL_PATH)
        conf            = self._get("confidence",            _DEFAULT_CONF)
        margin          = self._get("margin",                _DEFAULT_MARGIN)
        overlap_thr     = self._get("overlap_threshold",     _DEFAULT_OVERLAP_THR)
        alarm_threshold = self._get("alarm_frame_threshold", _DEFAULT_ALARM_THRESHOLD)
        alert_hold_secs = self._get("alert_hold_seconds",    _DEFAULT_ALERT_HOLD_SECS)

        pose_model_path = self._get("pose_model_path", _DEFAULT_POSE_MODEL_PATH)

        model      = YOLO(model_path)
        pose_model = load_pose_model(pose_model_path)

        cap         = cv2.VideoCapture(video_path)
        fps         = cap.get(cv2.CAP_PROP_FPS) or 25
        hold_frames = int(alert_hold_secs * fps)

        frame_annotations: list[FrameAnnotation] = []

        # Counters for summary
        compliant_count = 0
        no_helmet_count = 0
        no_vest_count   = 0
        no_ppe_count    = 0

        # Alarm state
        non_compliance_persistence = 0
        ppe_alarm_active           = False
        last_non_compliant_frame   = -1

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

            detections: list[Detection] = []

            persons = []   # list[(box, conf)]
            helmets = []   # list[box]
            vests   = []   # list[box]

            # Parse YOLO detections into persons / helmets / vests
            for r in results:
                names = r.names
                boxes = r.boxes

                if boxes is None or len(boxes) == 0:
                    continue

                xyxy    = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs   = boxes.conf.cpu().numpy()

                for box, cls_id, conf_val in zip(xyxy, cls_ids, confs):
                    label = names[int(cls_id)]
                    b = box_xyxy(box)

                    if label == PERSON_CLS:
                        persons.append((b, conf_val))
                    elif label == HELMET_CLS:
                        helmets.append(b)
                    elif label == VEST_CLS:
                        vests.append(b)

            # Pose-verify each person box — rejects burning objects / vehicles
            if persons:
                pose_results = run_pose(pose_model, frame)
                persons = [
                    (b, c) for b, c in persons
                    if human_keypoints_in_box(pose_results, int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                ]

            for pbox, pconf in persons:
                status, color, expanded = raw_status(
                    pbox, helmets, vests,
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

            non_compliant_this_frame = any(
                d.label in ("NO HELMET", "NO VEST", "NO PPE")
                for d in detections
            )

            if non_compliant_this_frame:
                non_compliance_persistence += 1
                last_non_compliant_frame = frame_idx

                if (non_compliance_persistence >= alarm_threshold
                        and not ppe_alarm_active):
                    ppe_alarm_active = True
                    if job_id:
                        self._save_alert(
                            frame,
                            "ppe_non_compliance",
                            job_id,
                            frame_idx,
                            extra={
                                "persistence": non_compliance_persistence,
                                "statuses": list({d.label for d in detections}),
                            },
                        )
            else:
                non_compliance_persistence = max(0, non_compliance_persistence - 1)
                if ppe_alarm_active and (frame_idx - last_non_compliant_frame) > hold_frames:
                    ppe_alarm_active = False
                    non_compliance_persistence = 0

            status_lines: list[str] = []
            if ppe_alarm_active:
                status_lines.append("!! PPE NON-COMPLIANCE ALARM")
            elif non_compliant_this_frame:
                status_lines.append("PPE NON-COMPLIANCE DETECTED")

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
                "compliant_person_frames":   compliant_count,
                "no_helmet_person_frames":   no_helmet_count,
                "no_vest_person_frames":     no_vest_count,
                "no_ppe_person_frames":      no_ppe_count,
                "total_frames":              frame_idx,
                "device":                    device,
            },
        )