import cv2
import numpy as np
import supervision as sv
from collections import defaultdict
from ultralytics import YOLO

from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ..pose_utils import _DEFAULT_POSE_MODEL_PATH, load_pose_model, run_pose, human_keypoints_in_box
from ...config import settings


_DEFAULT_MODEL_PATH      = "app/models/floating.pt"
_DEFAULT_CONF            = 0.30
_DEFAULT_ALARM_SECS      = 1.0
_DEFAULT_ALARM_HOLD_SECS = 4.0


# 0 -> floating_object (no keypoint check)
# 1 -> human (keypoint check required)
_CLS_FLOATING_OBJECT = 0
_CLS_HUMAN           = 1


def class_name_from_id(cls_id: int) -> str:
    if cls_id == _CLS_FLOATING_OBJECT:
        return "floating_object"
    if cls_id == _CLS_HUMAN:
        return "human"
    return "unknown"


@register_kpi
class FloatingKPI(BaseKPI):
    name = "floating"
    display_name = "Floating Object / Person"
    color = (255, 0, 0)

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path      = self._get("model_path",           _DEFAULT_MODEL_PATH)
        pose_model_path = self._get("pose_model_path",      _DEFAULT_POSE_MODEL_PATH)
        conf            = self._get("confidence",           _DEFAULT_CONF)
        alarm_secs      = self._get("alarm_seconds",        _DEFAULT_ALARM_SECS)
        alarm_hold_secs = self._get("alert_hold_seconds",   _DEFAULT_ALARM_HOLD_SECS)

        model      = YOLO(model_path)
        pose_model = load_pose_model(pose_model_path)

        tracker = sv.ByteTrack()

        video_info   = sv.VideoInfo.from_video_path(video_path)
        fps          = video_info.fps or 25
        alarm_frames = int(alarm_secs * fps)
        hold_frames  = int(alarm_hold_secs * fps)

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

        track_frame_counts: dict[int, int] = defaultdict(int)
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

            detections: list[Detection] = []
            labels: list[str] = []

            valid_xyxy = []
            valid_conf = []
            valid_cls  = []

            if len(sv_detections) > 0:
                human_candidates = []
                pose_needed_indices = []

                for i in range(len(sv_detections)):
                    cls_id = int(sv_detections.class_id[i]) if sv_detections.class_id is not None else -1
                    conf_val = float(sv_detections.confidence[i]) if sv_detections.confidence is not None else conf
                    box = sv_detections.xyxy[i]

                    if cls_id == _CLS_FLOATING_OBJECT:
                        valid_xyxy.append(box)
                        valid_conf.append(conf_val)
                        valid_cls.append(cls_id)

                    elif cls_id == _CLS_HUMAN:
                        human_candidates.append(box)
                        pose_needed_indices.append((box, conf_val, cls_id))

                if pose_needed_indices:
                    pose_results = run_pose(pose_model, frame)
                    for box, conf_val, cls_id in pose_needed_indices:
                        x1, y1, x2, y2 = map(int, box)
                        if human_keypoints_in_box(pose_results, x1, y1, x2, y2):
                            valid_xyxy.append(box)
                            valid_conf.append(conf_val)
                            valid_cls.append(cls_id)

            if valid_xyxy:
                valid_detections = sv.Detections(
                    xyxy=np.array(valid_xyxy, dtype=np.float32),
                    confidence=np.array(valid_conf, dtype=np.float32),
                    class_id=np.array(valid_cls, dtype=np.int32),
                )
                tracked_detections = tracker.update_with_detections(valid_detections)
            else:
                tracked_detections = sv.Detections.empty()

            annotated_xyxy = []
            annotated_conf = []
            annotated_cls  = []

            if len(tracked_detections) > 0:
                for i in range(len(tracked_detections)):
                    x1, y1, x2, y2 = map(int, tracked_detections.xyxy[i])
                    conf_val = float(tracked_detections.confidence[i]) if tracked_detections.confidence is not None else conf
                    cls_id = int(tracked_detections.class_id[i]) if tracked_detections.class_id is not None else -1
                    label_name = class_name_from_id(cls_id)

                    tracker_id = None
                    if tracked_detections.tracker_id is not None and len(tracked_detections.tracker_id) > i:
                        tracker_id = int(tracked_detections.tracker_id[i])

                    detections.append(
                        Detection(x1, y1, x2, y2, label_name, conf_val)
                    )

                    if tracker_id is not None:
                        track_frame_counts[tracker_id] += 1
                        visible_secs = track_frame_counts[tracker_id] / fps
                        is_alarm = track_frame_counts[tracker_id] >= alarm_frames

                        if is_alarm and tracker_id not in alarmed_ids:
                            alarmed_ids.add(tracker_id)
                            alarm_hold_until[tracker_id] = frame_idx + hold_frames

                            if job_id:
                                self._save_alert(
                                    frame,
                                    "floating_alarm",
                                    job_id,
                                    frame_idx,
                                    extra={
                                        "tracker_id": int(tracker_id),
                                        "visible_secs": float(visible_secs),
                                        "class": label_name,
                                    },
                                )

                        if tracker_id in alarmed_ids:
                            alarm_hold_until[tracker_id] = frame_idx + hold_frames

                        label = f"{label_name} #{tracker_id} {visible_secs:.1f}s"
                        if is_alarm:
                            label += " ALARM"
                    else:
                        label = label_name

                    labels.append(label)
                    annotated_xyxy.append([x1, y1, x2, y2])
                    annotated_conf.append(conf_val)
                    annotated_cls.append(cls_id)

            alarm_active = any(
                frame_idx <= alarm_hold_until.get(tid, -1)
                for tid in alarmed_ids
            )

            status_lines: list[str] = []
            if alarm_active:
                status_lines.append("!! FLOATING OBJECT / PERSON ALARM")

            if annotated_xyxy:
                annotated_sv = sv.Detections(
                    xyxy=np.array(annotated_xyxy, dtype=np.float32),
                    confidence=np.array(annotated_conf, dtype=np.float32),
                    class_id=np.array(annotated_cls, dtype=np.int32),
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

            if alarm_active:
                cv2.putText(
                    annotated_frame,
                    "!! FLOATING OBJECT / PERSON ALARM",
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

        frames_with_floating = sum(
            1 for fa in frame_annotations
            if any(d.label == "floating_object" for d in fa.detections)
        )
        frames_with_human = sum(
            1 for fa in frame_annotations
            if any(d.label == "human" for d in fa.detections)
        )

        return KPIResult(
            kpi_name=self.name,
            display_name=self.display_name,
            color=self.color,
            frame_annotations=frame_annotations,
            summary={
                "frames_with_floating": frames_with_floating,
                "frames_with_human":    frames_with_human,
                "alarm_triggered":      len(alarmed_ids) > 0,
                "unique_alarm_tracks":  len(alarmed_ids),
                "total_frames":         frame_idx,
                "device":               device,
            },
        )