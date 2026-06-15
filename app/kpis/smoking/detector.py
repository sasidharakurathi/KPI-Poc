import cv2
import supervision as sv
from collections import defaultdict
from ultralytics import YOLO

from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ...config import settings


_DEFAULT_MODEL_PATH      = "app/models/smoking.pt"
_DEFAULT_CONF            = 0.40
_DEFAULT_ALARM_SECS      = 1.0
_DEFAULT_ALARM_HOLD_SECS = 4.0


@register_kpi
class SmokingKPI(BaseKPI):
    name         = "smoking"
    display_name = "Smoking"
    color        = (0, 255, 255)

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path      = self._get("model_path",         _DEFAULT_MODEL_PATH)
        conf            = self._get("confidence",         _DEFAULT_CONF)
        alarm_secs      = self._get("alarm_seconds",      _DEFAULT_ALARM_SECS)
        alarm_hold_secs = self._get("alert_hold_seconds", _DEFAULT_ALARM_HOLD_SECS)

        model = YOLO(model_path)

        # ── Supervision annotators ──────────────────────────────────────────
        box_annotator   = sv.BoxAnnotator(color=sv.Color.from_bgr_tuple(self.color))
        label_annotator = sv.LabelAnnotator(color=sv.Color.from_bgr_tuple(self.color))

        # ── Tracker ─────────────────────────────────────────────────────────
        tracker = sv.ByteTrack()

        # ── Video info ──────────────────────────────────────────────────────
        video_info   = sv.VideoInfo.from_video_path(video_path)
        fps          = video_info.fps or 25
        alarm_frames = int(alarm_secs * fps)
        hold_frames  = int(alarm_hold_secs * fps)

        cap = cv2.VideoCapture(video_path)

        frame_annotations: list[FrameAnnotation] = []

        # ── Per-track state ─────────────────────────────────────────────────
        track_frame_counts: dict[int, int] = defaultdict(int)
        track_last_seen:    dict[int, int] = defaultdict(int)
        alarmed_ids:        set[int]       = set()
        alarm_hold_until:   dict[int, int] = {}

        smoke_alarm_active = False
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ── Inference ───────────────────────────────────────────────────
            results = model.predict(
                source=frame,
                device=device,
                half=half,
                conf=conf,
                verbose=False,
            )

            # ── Convert to sv.Detections ────────────────────────────────────
            sv_detections = sv.Detections.from_ultralytics(results[0])

            # ── Track ───────────────────────────────────────────────────────
            if len(sv_detections) > 0:
                sv_detections = tracker.update_with_detections(sv_detections)

            # ── Per-track persistence logic ─────────────────────────────────
            detections: list[Detection] = []
            labels:     list[str]       = []

            if len(sv_detections) > 0 and sv_detections.tracker_id is not None:
                for i, tracker_id in enumerate(sv_detections.tracker_id):
                    if tracker_id is None:
                        continue

                    x1, y1, x2, y2 = map(int, sv_detections.xyxy[i])
                    conf_val = (
                        float(sv_detections.confidence[i])
                        if sv_detections.confidence is not None
                        else conf
                    )

                    track_frame_counts[tracker_id] += 1
                    track_last_seen[tracker_id]     = frame_idx

                    visible_secs = track_frame_counts[tracker_id] / fps
                    is_alarm     = track_frame_counts[tracker_id] >= alarm_frames

                    if is_alarm and tracker_id not in alarmed_ids:
                        alarmed_ids.add(tracker_id)
                        alarm_hold_until[tracker_id] = frame_idx + hold_frames
                        if job_id:
                            self._save_alert(
                                frame,
                                "smoking_alarm",
                                job_id,
                                frame_idx,
                                extra={
                                    "tracker_id":   int(tracker_id),
                                    "visible_secs": float(visible_secs),
                                },
                            )

                    # extend hold window while cigarette is still visible
                    if tracker_id in alarmed_ids:
                        alarm_hold_until[tracker_id] = frame_idx + hold_frames

                    label = f"#{int(tracker_id)} {visible_secs:.1f}s"
                    if is_alarm:
                        label += " ALARM"
                    labels.append(label)

                    detections.append(Detection(x1, y1, x2, y2, "smoking", conf_val))

            # ── Update global alarm active flag ─────────────────────────────
            smoke_alarm_active = any(
                frame_idx <= alarm_hold_until.get(tid, -1)
                for tid in alarmed_ids
            )

            # ── Annotate frame ──────────────────────────────────────────────
            annotated = frame.copy()
            if len(sv_detections) > 0:
                annotated = box_annotator.annotate(
                    scene=annotated, detections=sv_detections
                )
                if labels:
                    annotated = label_annotator.annotate(
                        scene=annotated, detections=sv_detections, labels=labels
                    )

            if smoke_alarm_active:
                cv2.putText(
                    annotated,
                    "!! SMOKING ALARM ACTIVE",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # ── Build FrameAnnotation ───────────────────────────────────────
            status_lines: list[str] = []
            if smoke_alarm_active:
                status_lines.append("!! SMOKING ALARM ACTIVE")

            frame_annotations.append(
                FrameAnnotation(
                    frame_idx=frame_idx,
                    detections=detections,
                    status_lines=status_lines,
                )
            )
            frame_idx += 1

        cap.release()

        frames_with_smoking = sum(
            1 for fa in frame_annotations
            if any(d.label == "smoking" for d in fa.detections)
        )

        return KPIResult(
            kpi_name=self.name,
            display_name=self.display_name,
            color=self.color,
            frame_annotations=frame_annotations,
            summary={
                "frames_with_smoking":  frames_with_smoking,
                "alarm_triggered":      len(alarmed_ids) > 0,
                "unique_smokers_found": len(alarmed_ids),
                "total_frames":         frame_idx,
                "device":               device,
            },
        )