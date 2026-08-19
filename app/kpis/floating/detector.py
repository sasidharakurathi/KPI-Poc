import cv2
import numpy as np
import supervision as sv
from collections import defaultdict
from ultralytics import YOLO

from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ..pose_utils import _DEFAULT_POSE_MODEL_PATH, load_pose_model, run_pose, human_keypoints_in_box
from ...config import settings

_CLS_FLOATING_OBJECT = 0
_CLS_HUMAN           = 1


@register_kpi
class FloatingKPI(BaseKPI):
    name = "floating"
    display_name = "Floating Object / Person"

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path      = self._get("model_path",         "app/models/floating.pt")
        pose_model_path = self._get("pose_model_path",    _DEFAULT_POSE_MODEL_PATH)
        conf            = self._get("confidence",         0.30)
        alarm_secs      = self._get("alarm_seconds",      1.0)
        alert_hold_secs = self._get("alert_hold_seconds", 4.0)

        model      = YOLO(model_path)
        pose_model = load_pose_model(pose_model_path)
        tracker    = sv.ByteTrack()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        alarm_frames = int(alarm_secs * fps)

        track_counts: dict[int, int] = defaultdict(int)
        alarmed_ids:  set[int]       = set()
        alert_events = 0
        frame_idx    = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self._observe(frame, frame_idx, job_id)

            results = model.predict(frame, conf=conf, device=device, half=half, verbose=False)
            sv_dets = sv.Detections.from_ultralytics(results[0])

            valid_xyxy, valid_conf, valid_cls = [], [], []

            if len(sv_dets) > 0:
                human_candidates = []
                for i in range(len(sv_dets)):
                    cls_id   = int(sv_dets.class_id[i]) if sv_dets.class_id is not None else -1
                    conf_val = float(sv_dets.confidence[i]) if sv_dets.confidence is not None else conf
                    box      = sv_dets.xyxy[i]
                    if cls_id == _CLS_FLOATING_OBJECT:
                        valid_xyxy.append(box); valid_conf.append(conf_val); valid_cls.append(cls_id)
                    elif cls_id == _CLS_HUMAN:
                        human_candidates.append((box, conf_val, cls_id))

                if human_candidates:
                    pose_results = run_pose(pose_model, frame)
                    for box, conf_val, cls_id in human_candidates:
                        x1, y1, x2, y2 = map(int, box)
                        if human_keypoints_in_box(pose_results, x1, y1, x2, y2):
                            valid_xyxy.append(box); valid_conf.append(conf_val); valid_cls.append(cls_id)

            if valid_xyxy:
                valid_sv = sv.Detections(
                    xyxy=np.array(valid_xyxy, dtype=np.float32),
                    confidence=np.array(valid_conf, dtype=np.float32),
                    class_id=np.array(valid_cls, dtype=np.int32),
                )
                tracked = tracker.update_with_detections(valid_sv)
            else:
                tracked = sv.Detections.empty()

            for i in range(len(tracked)):
                x1, y1, x2, y2 = map(int, tracked.xyxy[i])
                cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
                cname  = "floating_object" if cls_id == _CLS_FLOATING_OBJECT else "human"
                conf_val = float(tracked.confidence[i]) if tracked.confidence is not None else conf
                tracker_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None

                if tracker_id is None:
                    continue

                track_counts[tracker_id] += 1
                if track_counts[tracker_id] >= alarm_frames and tracker_id not in alarmed_ids:
                    alarmed_ids.add(tracker_id)
                    alert_events += 1
                    self._save_alert(
                        "floating_alarm", job_id, frame_idx,
                        confidence=conf_val,
                        extra={"tracker_id": tracker_id, "class": cname},
                        boxes=[(x1, y1, x2, y2, f"{cname} #{tracker_id}", (255, 0, 0))],
                    )

            frame_idx += 1

        cap.release()
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "alert_events":       alert_events,
            "unique_alarm_tracks": len(alarmed_ids),
            "total_frames":       frame_idx,
            "device":             device,
        })
