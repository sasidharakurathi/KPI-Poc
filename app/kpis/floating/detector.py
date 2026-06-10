import cv2
from ultralytics import YOLO

from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ..pose_utils import _DEFAULT_POSE_MODEL_PATH, load_pose_model, run_pose, human_keypoints_in_box
from ...config import settings

_DEFAULT_MODEL_PATH      = "app/models/floating.pt"
_DEFAULT_CONF            = 0.40
_DEFAULT_ALARM_THRESHOLD = 10
_DEFAULT_ALARM_HOLD_SECS = 3.0

# 0 -> floating_object (no keypoint check — life ring / debris is a real alert)
# 1 -> human           (keypoint check required to reject burning objects/vehicles)
_CLS_FLOATING_OBJECT = 0
_CLS_HUMAN           = 1


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
        alarm_threshold = self._get("alarm_frame_threshold", _DEFAULT_ALARM_THRESHOLD)
        alarm_hold_secs = self._get("alert_hold_seconds",   _DEFAULT_ALARM_HOLD_SECS)

        model      = YOLO(model_path)
        pose_model = load_pose_model(pose_model_path)

        cap         = cv2.VideoCapture(video_path)
        fps         = cap.get(cv2.CAP_PROP_FPS) or 25
        hold_frames = int(alarm_hold_secs * fps)

        frame_annotations: list[FrameAnnotation] = []

        detection_persistence = 0
        alarm_active          = False
        last_detection_frame  = -1

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

            # Separate human candidates (need pose check) from object candidates
            human_candidates: list[tuple[int, int, int, int, float]] = []
            object_detections: list[Detection] = []

            for r in results:
                for box in r.boxes:
                    cls_id   = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if cls_id == _CLS_FLOATING_OBJECT:
                        object_detections.append(
                            Detection(x1, y1, x2, y2, "floating_object", conf_val)
                        )
                    elif cls_id == _CLS_HUMAN:
                        human_candidates.append((x1, y1, x2, y2, conf_val))

            # Verify human candidates with pose model
            verified_human_detections: list[Detection] = []
            if human_candidates:
                pose_results = run_pose(pose_model, frame)
                for x1, y1, x2, y2, conf_val in human_candidates:
                    if human_keypoints_in_box(pose_results, x1, y1, x2, y2):
                        verified_human_detections.append(
                            Detection(x1, y1, x2, y2, "human", conf_val)
                        )

            detections = object_detections + verified_human_detections
            any_detection_this_frame = len(detections) > 0

            if any_detection_this_frame:
                detection_persistence += 1
                last_detection_frame = frame_idx

                if detection_persistence >= alarm_threshold and not alarm_active:
                    alarm_active = True
                    if job_id:
                        self._save_alert(
                            frame,
                            "floating_alarm",
                            job_id,
                            frame_idx,
                            extra={
                                "persistence": detection_persistence,
                                "classes": list({d.label for d in detections}),
                            },
                        )
            else:
                detection_persistence = max(0, detection_persistence - 1)
                if alarm_active and (frame_idx - last_detection_frame) > hold_frames:
                    alarm_active = False
                    detection_persistence = 0

            status_lines: list[str] = []
            if alarm_active:
                status_lines.append("!! FLOATING OBJECT / PERSON ALARM")

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
                "alarm_triggered":      frames_with_floating > 0 or frames_with_human > 0,
                "total_frames":         frame_idx,
                "device":               device,
            },
        )
