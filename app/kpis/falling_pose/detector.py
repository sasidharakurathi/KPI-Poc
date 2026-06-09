import cv2
from collections import deque
from ultralytics import YOLO

from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ...config import settings

_DEFAULT_MODEL_PATH      = "app/models/falling-pose.pt"
_DEFAULT_POSE_MODEL_PATH = "app/models/yolo26s-pose.pt"
_DEFAULT_CONF            = 0.30
_DEFAULT_TRIGGER_RATIO   = 0.80
_DEFAULT_WINDOW_SECS     = 1.0
_DEFAULT_ALERT_HOLD_SECS = 2.0

# Model class IDs
_CLS_STANDING = 0
_CLS_FALLING  = 1
_CLS_FALLEN   = 2

_COLOR_FALLING = (0, 0, 255)    # red
_COLOR_FALLEN  = (0, 140, 255)  # orange-red

_CLASS_COLORS = {
    _CLS_FALLING: _COLOR_FALLING,
    _CLS_FALLEN:  _COLOR_FALLEN,
}

# Shoulder and hip keypoint indices (COCO-17 format)
_KEY_INDICES = [5, 6, 11, 12]


def _human_keypoints_in_box(pose_results, x1: int, y1: int, x2: int, y2: int) -> bool:
    """
    Returns True if any person detected by the pose model has shoulder/hip
    keypoints whose centroid lies inside the candidate fall bounding box.
    A burning scooter or vehicle produces zero human keypoints, so it fails.
    A real fallen person has shoulder/hip keypoints inside the box, so it passes.
    """
    for r in pose_results:
        if r.keypoints is None:
            continue
        # kps shape: (N_persons, 17, 2) — pixel xy coordinates
        kps = r.keypoints.xy.cpu().numpy()
        for person_kps in kps:
            valid_pts = [
                (float(person_kps[i][0]), float(person_kps[i][1]))
                for i in _KEY_INDICES
                if person_kps[i][0] > 0 and person_kps[i][1] > 0
            ]
            if not valid_pts:
                continue

            # Centroid of the valid shoulder/hip keypoints
            cx = sum(p[0] for p in valid_pts) / len(valid_pts)
            cy = sum(p[1] for p in valid_pts) / len(valid_pts)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True

            # Fallback: any individual keypoint inside the box
            for kx, ky in valid_pts:
                if x1 <= kx <= x2 and y1 <= ky <= y2:
                    return True

    return False


@register_kpi
class FallingPoseKPI(BaseKPI):
    name         = "falling_pose"
    display_name = "Falling Pose"
    color        = _COLOR_FALLING

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path      = self._get("model_path",        _DEFAULT_MODEL_PATH)
        pose_model_path = self._get("pose_model_path",   _DEFAULT_POSE_MODEL_PATH)
        conf            = self._get("confidence",         _DEFAULT_CONF)
        trigger_ratio   = self._get("trigger_ratio",      _DEFAULT_TRIGGER_RATIO)
        window_secs     = self._get("window_seconds",     _DEFAULT_WINDOW_SECS)
        alert_hold_secs = self._get("alert_hold_seconds", _DEFAULT_ALERT_HOLD_SECS)

        model      = YOLO(model_path)
        pose_model = YOLO(pose_model_path)

        cap         = cv2.VideoCapture(video_path)
        fps         = cap.get(cv2.CAP_PROP_FPS) or 25
        window_size = max(1, int(window_secs * fps))
        hold_frames = int(alert_hold_secs * fps)

        detect_window   = deque(maxlen=window_size)
        alert_countdown = 0
        in_fall_event   = False
        fall_events     = 0

        frame_annotations: list[FrameAnnotation] = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                source=frame,
                conf=conf,
                device=device,
                half=half,
                verbose=False,
            )

            # Collect candidate fall boxes before pose verification
            candidates: list[tuple[int, int, int, int, int, float]] = []  # (x1,y1,x2,y2,cls_id,conf)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == _CLS_STANDING:
                        continue
                    conf_val = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    candidates.append((x1, y1, x2, y2, cls_id, conf_val))

            detections: list[Detection] = []
            frame_has_fall = False

            if candidates:
                # Run pose model only when there are candidate fall boxes
                pose_results = pose_model.predict(
                    source=frame,
                    device=device,
                    half=half,
                    verbose=False,
                )

                for x1, y1, x2, y2, cls_id, conf_val in candidates:
                    if not _human_keypoints_in_box(pose_results, x1, y1, x2, y2):
                        continue  # no human keypoints inside box — reject

                    cls_name = model.names[cls_id]
                    color    = _CLASS_COLORS.get(cls_id, self.color)
                    detections.append(
                        Detection(x1, y1, x2, y2, cls_name.upper(), conf_val, color=color)
                    )
                    frame_has_fall = True

            # Sliding-window vote
            detect_window.append(1 if frame_has_fall else 0)
            window_full  = len(detect_window) == window_size
            window_ratio = sum(detect_window) / window_size if window_full else 0.0
            is_sustained = window_full and window_ratio >= trigger_ratio

            if is_sustained:
                alert_countdown = hold_frames
                if not in_fall_event:
                    in_fall_event = True
                    fall_events  += 1
                    if job_id:
                        self._save_alert(
                            frame,
                            "fall_detected",
                            job_id,
                            frame_idx,
                            extra={
                                "fall_event_number": fall_events,
                                "window_ratio": round(window_ratio, 2),
                            },
                        )
            else:
                if alert_countdown > 0:
                    alert_countdown -= 1
                if window_full and window_ratio < 0.3:
                    in_fall_event = False

            alert_active = is_sustained or alert_countdown > 0

            status_lines: list[str] = []
            if alert_active:
                status_lines.append("!! FALL DETECTED - ALERT!")
                status_lines.append(f"Fall events: {fall_events}")

            frame_annotations.append(FrameAnnotation(
                frame_idx=frame_idx,
                detections=detections if alert_active else [],
                status_lines=status_lines,
            ))
            frame_idx += 1

        cap.release()

        falling_frames = sum(
            1 for fa in frame_annotations
            if any(d.label in ("FALLING", "FALLEN") for d in fa.detections)
        )

        return KPIResult(
            kpi_name=self.name,
            display_name=self.display_name,
            color=self.color,
            frame_annotations=frame_annotations,
            summary={
                "fall_events":    fall_events,
                "falling_frames": falling_frames,
                "total_frames":   frame_idx,
                "device":         device,
            },
        )
