import cv2
import numpy as np
import supervision as sv
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult, get_dynamic_scale
from ..registry import register_kpi
from ...config import settings

_DEFAULT_WORKER_MODEL_PATH = r"D:\jana-poc\app\models\uniform.pt"
_DEFAULT_PERSON_MODEL_PATH = r"D:\jana-poc\app\models\person yolo26s.pt"

_DEFAULT_CONF              = 0.25
_DEFAULT_MARGIN            = 0.10
_DEFAULT_MATCH_IOU_THR     = 0.30
_DEFAULT_ALARM_SECS        = 2.0
_DEFAULT_ALERT_HOLD_SECS   = 4.0
_DEFAULT_TRACK_TTL_SECS    = 3.0

# --- Color and Reflective Strap Thresholds ---
_DEFAULT_ORANGE_H_MIN      = 5     # Excludes red uniforms (Hue 0-5 and 170-180)
_DEFAULT_ORANGE_H_MAX      = 22    # OpenCV Hue max for orange before yellow
_DEFAULT_ORANGE_S_MIN      = 100   # Filters out dull/grey clothing
_DEFAULT_ORANGE_V_MIN      = 50    # Minimum brightness for orange fabric
_DEFAULT_STRAP_V_MIN       = 190   # High brightness floor for reflective silver/white tape
_DEFAULT_STRAP_S_MAX       = 60    # Low saturation ceiling for silver/white tape
_DEFAULT_STRAP_MIN_FRAC    = 0.025 # Min percentage of upper-torso that must be reflective tape
_DEFAULT_TORSO_FRAC        = 0.40  # Top 40% of bounding box for chest/shoulder straps
_DEFAULT_LOWER_FRAC        = 0.75  # Lower 50%-75% of bounding box to verify coverall legs

_DEFAULT_SMOOTH_WINDOW_SECS = 0.5

PERSON_CLS = "person"
WORKER_CLS = "worker"


def box_xyxy(box):
    return [float(x) for x in box]


def expand_box(box, margin=0.10):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return [x1 - margin * w, y1 - margin * h, x2 + margin * w, y2 + margin * h]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_persons_to_workers(person_boxes, worker_boxes, margin=0.10, iou_thr=0.30):
    if not person_boxes or not worker_boxes:
        return {}

    n_p, n_w = len(person_boxes), len(worker_boxes)
    cost = np.ones((n_p, n_w), dtype=np.float32)
    for i, pbox in enumerate(person_boxes):
        expanded = expand_box(pbox, margin)
        for j, wbox in enumerate(worker_boxes):
            cost[i, j] = 1.0 - iou(expanded, wbox)

    row_idx, col_idx = linear_sum_assignment(cost)

    matches = {}
    for r, c in zip(row_idx, col_idx):
        matched_iou = 1.0 - cost[r, c]
        if matched_iou >= iou_thr:
            matches[int(r)] = int(c)
    return matches


def ppe_color_score(
    frame,
    box,
    orange_h_bounds=(_DEFAULT_ORANGE_H_MIN, _DEFAULT_ORANGE_H_MAX),
    orange_s_min=_DEFAULT_ORANGE_S_MIN,
    orange_v_min=_DEFAULT_ORANGE_V_MIN,
    strap_v_min=_DEFAULT_STRAP_V_MIN,
    strap_s_max=_DEFAULT_STRAP_S_MAX,
    strap_min_frac=_DEFAULT_STRAP_MIN_FRAC,
    torso_frac=_DEFAULT_TORSO_FRAC,
    lower_frac=_DEFAULT_LOWER_FRAC,
):
    """
    Dual-check validation:
    1. Verifies strict orange HSV color presence in BOTH upper torso and legs to rule out 
       red coveralls and standalone orange t-shirts/jackets.
    2. Verifies high-brightness, low-saturation reflective silver/white straps on upper torso.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h_img, w_img = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    if x2 <= x1 or y2 <= y1:
        return False, 0.0

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False, 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    box_h = y2 - y1

    # Define upper-torso region (chest/shoulders)
    torso_mask = np.zeros(h.shape[:2], dtype=bool)
    torso_mask[: max(1, int(box_h * torso_frac)), :] = True

    # Define lower-body region (legs/lower coverall)
    legs_mask = np.zeros(h.shape[:2], dtype=bool)
    legs_start = int(box_h * 0.50)
    legs_end = max(legs_start + 1, int(box_h * lower_frac))
    legs_mask[legs_start:legs_end, :] = True

    # 1. Strict Orange Mask (Hue 5 to 22 excludes red coveralls at 0-5 and 170-180)
    orange_mask = (
        (h >= orange_h_bounds[0])
        & (h <= orange_h_bounds[1])
        & (s >= orange_s_min)
        & (v >= orange_v_min)
    )

    torso_orange_ratio = float(orange_mask[torso_mask].sum()) / float(max(1, torso_mask.sum()))
    legs_orange_ratio = float(orange_mask[legs_mask].sum()) / float(max(1, legs_mask.sum()))

    # 2. Reflective Silver/White Strap Mask
    strap_mask = (v >= strap_v_min) & (s <= strap_s_max)
    strap_ratio = float(strap_mask[torso_mask].sum()) / float(max(1, torso_mask.sum()))

    # Must be orange on top AND bottom (full coverall suit, not just a shirt)
    is_full_orange = (torso_orange_ratio >= 0.15) and (legs_orange_ratio >= 0.10)

    # Must have silver reflective straps
    has_reflective_straps = strap_ratio >= strap_min_frac

    return (is_full_orange and has_reflective_straps), strap_ratio


@register_kpi
class UniformKPI(BaseKPI):
    name = "Uniform_detection"
    display_name = "Uniform Compliance"
    color = (0, 165, 255)

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        worker_model_path = self._get("worker_model_path", _DEFAULT_WORKER_MODEL_PATH)
        person_model_path = self._get("person_model_path", _DEFAULT_PERSON_MODEL_PATH)
        conf              = self._get("confidence",          _DEFAULT_CONF)
        margin            = self._get("margin",              _DEFAULT_MARGIN)
        match_iou_thr     = self._get("match_iou_threshold", _DEFAULT_MATCH_IOU_THR)
        alarm_secs        = self._get("alarm_seconds",       _DEFAULT_ALARM_SECS)
        alert_hold_secs   = self._get("alert_hold_seconds",  _DEFAULT_ALERT_HOLD_SECS)
        track_ttl_secs    = self._get("track_ttl_seconds",   _DEFAULT_TRACK_TTL_SECS)

        orange_h_min      = self._get("orange_h_min",        _DEFAULT_ORANGE_H_MIN)
        orange_h_max      = self._get("orange_h_max",        _DEFAULT_ORANGE_H_MAX)
        orange_s_min      = self._get("orange_s_min",        _DEFAULT_ORANGE_S_MIN)
        orange_v_min      = self._get("orange_v_min",        _DEFAULT_ORANGE_V_MIN)
        strap_v_min       = self._get("strap_v_min",         _DEFAULT_STRAP_V_MIN)
        strap_s_max       = self._get("strap_s_max",         _DEFAULT_STRAP_S_MAX)
        strap_min_frac    = self._get("strap_min_frac",      _DEFAULT_STRAP_MIN_FRAC)
        torso_frac        = self._get("torso_frac",          _DEFAULT_TORSO_FRAC)
        lower_frac        = self._get("lower_frac",          _DEFAULT_LOWER_FRAC)
        smooth_window_secs= self._get("smooth_window_secs",  _DEFAULT_SMOOTH_WINDOW_SECS)

        try:
            worker_model = YOLO(worker_model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load worker (uniform) YOLO model from '{worker_model_path}': {e}"
            ) from e

        try:
            person_model = YOLO(person_model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load person YOLO model from '{person_model_path}': {e}"
            ) from e

        tracker = sv.ByteTrack()

        video_info   = sv.VideoInfo.from_video_path(video_path)
        fps          = video_info.fps or 25
        alarm_frames = int(alarm_secs * fps)
        hold_frames  = int(alert_hold_secs * fps)
        track_ttl_frames = int(track_ttl_secs * fps)
        smooth_window_frames = max(1, int(round(smooth_window_secs * fps)))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = get_dynamic_scale(w, h) if w and h else 1.0

        box_annotator = sv.BoxAnnotator(thickness=max(1, int(round(2 * scale))))
        label_annotator = sv.LabelAnnotator(
            text_scale=0.5 * scale,
            text_thickness=max(1, int(round(1 * scale)))
        )

        person_names = person_model.names
        worker_names = worker_model.names

        person_cls_id = next((k for k, v in person_names.items() if v == PERSON_CLS), None)
        worker_cls_id = next((k for k, v in worker_names.items() if v == WORKER_CLS), None)

        if person_cls_id is None:
            raise RuntimeError(
                f"Person model '{person_model_path}' has no class named '{PERSON_CLS}'. "
                f"Available classes: {list(person_names.values())}"
            )
        if worker_cls_id is None:
            raise RuntimeError(
                f"Worker model '{worker_model_path}' has no class named '{WORKER_CLS}'. "
                f"Available classes: {list(worker_names.values())}"
            )

        frame_annotations: list[FrameAnnotation] = []

        uniform_count    = 0
        no_uniform_count = 0

        track_noncompliant_counts: dict[int, int] = defaultdict(int)
        track_last_seen: dict[int, int] = {}
        alarmed_ids: set[int] = set()
        alarm_hold_until: dict[int, int] = {}
        track_status_history: dict[int, deque] = {}

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            person_results = person_model.predict(
                source=frame,
                device=device,
                half=half,
                conf=conf,
                classes=[person_cls_id],
                verbose=False,
            )
            worker_results = worker_model.predict(
                source=frame,
                device=device,
                half=half,
                conf=conf,
                classes=[worker_cls_id],
                verbose=False,
            )

            persons_sv = sv.Detections.from_ultralytics(person_results[0])
            workers_sv = sv.Detections.from_ultralytics(worker_results[0])

            worker_boxes = [box_xyxy(b) for b in workers_sv.xyxy] if len(workers_sv) > 0 else []

            if len(persons_sv) > 0:
                tracked_persons_sv = tracker.update_with_detections(persons_sv)
            else:
                tracked_persons_sv = sv.Detections.empty()

            person_boxes = [box_xyxy(b) for b in tracked_persons_sv.xyxy] if len(tracked_persons_sv) > 0 else []

            # 1-to-1 Hungarian Matching between person detection and worker detection
            matches = match_persons_to_workers(
                person_boxes, worker_boxes, margin=margin, iou_thr=match_iou_thr
            )

            detections: list[Detection] = []
            labels: list[str] = []
            annotated_detections_xyxy = []
            annotated_detections_conf = []
            annotated_detections_class = []

            if len(tracked_persons_sv) > 0:
                for i in range(len(tracked_persons_sv)):
                    pbox = person_boxes[i]
                    pconf = float(tracked_persons_sv.confidence[i]) if tracked_persons_sv.confidence is not None else conf
                    tracker_id = None

                    if tracked_persons_sv.tracker_id is not None and len(tracked_persons_sv.tracker_id) > i:
                        tracker_id = int(tracked_persons_sv.tracker_id[i])

                    # Dual Gate Validation: Hungarian YOLO match AND HSV full-orange + strap verification
                    matched_worker = i in matches
                    is_orange_ppe, _ = ppe_color_score(
                        frame,
                        pbox,
                        orange_h_bounds=(orange_h_min, orange_h_max),
                        orange_s_min=orange_s_min,
                        orange_v_min=orange_v_min,
                        strap_v_min=strap_v_min,
                        strap_s_max=strap_s_max,
                        strap_min_frac=strap_min_frac,
                        torso_frac=torso_frac,
                        lower_frac=lower_frac,
                    )
                    raw_uniform = matched_worker and is_orange_ppe

                    # Temporal smoothing across recent frame history
                    if tracker_id is not None:
                        hist = track_status_history.get(tracker_id)
                        if hist is None:
                            hist = deque(maxlen=smooth_window_frames)
                            track_status_history[tracker_id] = hist
                        hist.append(1 if raw_uniform else 0)
                        smoothed_uniform = (sum(hist) / len(hist)) >= 0.5
                    else:
                        smoothed_uniform = raw_uniform

                    if smoothed_uniform:
                        status, color = "UNIFORM", (0, 180, 0)
                    else:
                        status, color = "NO UNIFORM", (0, 0, 255)

                    x1, y1, x2, y2 = map(int, pbox)

                    detections.append(
                        Detection(
                            x1, y1, x2, y2,
                            status,
                            float(pconf),
                            color=color,
                        )
                    )

                    if status == "UNIFORM":
                        uniform_count += 1
                    else:
                        no_uniform_count += 1

                    label = status
                    is_alarm = False
                    visible_secs = 0.0

                    if tracker_id is not None:
                        track_last_seen[tracker_id] = frame_idx

                        if status == "NO UNIFORM":
                            track_noncompliant_counts[tracker_id] += 1

                            visible_secs = track_noncompliant_counts[tracker_id] / fps
                            is_alarm = track_noncompliant_counts[tracker_id] >= alarm_frames

                            if is_alarm and tracker_id not in alarmed_ids:
                                alarmed_ids.add(tracker_id)
                                alarm_hold_until[tracker_id] = frame_idx + hold_frames

                                if job_id:
                                    self._save_alert(
                                        frame,
                                        "uniform_non_compliance",
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
                            track_noncompliant_counts[tracker_id] = 0
                            label = f"#{tracker_id} {status}"

                    labels.append(label)
                    annotated_detections_xyxy.append([x1, y1, x2, y2])
                    annotated_detections_conf.append(float(pconf))
                    annotated_detections_class.append(0)

            # Prune stale tracking data
            stale_ids = [
                tid for tid, last_seen in track_last_seen.items()
                if frame_idx - last_seen > track_ttl_frames
            ]
            for tid in stale_ids:
                track_last_seen.pop(tid, None)
                track_noncompliant_counts.pop(tid, None)
                alarm_hold_until.pop(tid, None)
                track_status_history.pop(tid, None)
                alarmed_ids.discard(tid)

            uniform_alarm_active = any(
                frame_idx <= alarm_hold_until.get(tid, -1)
                for tid in alarmed_ids
            )

            non_compliant_this_frame = any(
                d.label == "NO UNIFORM"
                for d in detections
            )

            status_lines: list[str] = []
            if uniform_alarm_active:
                status_lines.append("!! UNIFORM NON-COMPLIANCE ALARM")
            elif non_compliant_this_frame:
                status_lines.append("UNIFORM NON-COMPLIANCE DETECTED")

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

            if uniform_alarm_active:
                cv2.putText(
                    annotated_frame,
                    "!! UNIFORM NON-COMPLIANCE ALARM",
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
                "uniform_person_frames": uniform_count,
                "no_uniform_person_frames": no_uniform_count,
                "alarm_triggered": len(alarmed_ids) > 0,
                "total_frames": frame_idx,
                "device": device,
            },
        )