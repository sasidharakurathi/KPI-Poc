import cv2
from collections import deque
from ultralytics import YOLO

from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ...config import settings

_DEFAULT_PERSON_MODEL_PATH = settings.MOBILE_PERSON_MODEL_PATH
_DEFAULT_PHONE_MODEL_PATH  = settings.MOBILE_PHONE_MODEL_PATH
_DEFAULT_PERSON_CONF       = 0.5
_DEFAULT_PHONE_CONF        = 0.30
_DEFAULT_VOTE_WINDOW       = 20
_DEFAULT_VOTE_REQUIRED     = 6
_DEFAULT_MIN_PERSON_HEIGHT = 120
_DEFAULT_ROI_HEIGHT_RATIO  = 0.70
_DEFAULT_ROI_PAD_X_RATIO   = 0.10
_DEFAULT_PHONE_ZONE_RATIO  = 0.85
_DEFAULT_USAGE_HOLD_SECS   = 3.0

COLOR_PERSON = (0, 0, 220)
COLOR_PHONE  = (0, 140, 255)


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


@register_kpi
class MobileUsageKPI(BaseKPI):
    name = "mobile_usage"
    display_name = "Mobile Phone Usage"
    color = COLOR_PERSON

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        # ── Parameters from config.json (with fallbacks) ──────────────────
        person_model_path = self._get("person_model_path",  _DEFAULT_PERSON_MODEL_PATH)
        phone_model_path  = self._get("phone_model_path",   _DEFAULT_PHONE_MODEL_PATH)
        person_conf       = self._get("person_confidence",  _DEFAULT_PERSON_CONF)
        phone_conf        = self._get("phone_confidence",   _DEFAULT_PHONE_CONF)
        vote_window       = self._get("vote_window",        _DEFAULT_VOTE_WINDOW)
        vote_required     = self._get("vote_required",      _DEFAULT_VOTE_REQUIRED)
        min_person_height = self._get("min_person_height",  _DEFAULT_MIN_PERSON_HEIGHT)
        roi_height_ratio  = self._get("roi_height_ratio",   _DEFAULT_ROI_HEIGHT_RATIO)
        roi_pad_x_ratio   = self._get("roi_pad_x_ratio",    _DEFAULT_ROI_PAD_X_RATIO)
        phone_zone_ratio  = self._get("phone_zone_ratio",   _DEFAULT_PHONE_ZONE_RATIO)
        usage_hold_secs   = self._get("usage_hold_seconds", _DEFAULT_USAGE_HOLD_SECS)

        person_model = YOLO(person_model_path)
        phone_model  = YOLO(phone_model_path)

        cap          = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        hold_frames  = int(usage_hold_secs * fps)

        # ── Per-track state ───────────────────────────────────────────────
        # usage_votes     : sliding window of phone_confirmed booleans
        # session_start   : frame_count when the current usage session began (None = not active)
        # display_secs    : frozen display value; 0.0 until first session, resets each new session
        # last_stable_frame: last frame_count where stable_usage was True (drives hold window)
        # prev_stable     : stable_usage from previous frame
        usage_votes:       dict[int, deque]      = {}
        session_start:     dict[int, int | None] = {}
        display_secs:      dict[int, float]      = {}
        last_stable_frame: dict[int, int]        = {}
        prev_stable:       dict[int, bool]       = {}

        frame_annotations: list[FrameAnnotation] = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            detections: list[Detection] = []

            person_results = person_model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=person_conf,
                device=device,
                half=half,
                verbose=False,
            )
            result = person_results[0]

            if result.boxes.id is not None:
                boxes     = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    person_h = y2 - y1
                    person_w = x2 - x1

                    if person_h < min_person_height:
                        continue

                    # Crop upper-body ROI
                    pad_x = int(person_w * roi_pad_x_ratio)
                    rx1 = max(0, x1 + pad_x)
                    ry1 = max(0, y1)
                    rx2 = min(frame_width,  x2 - pad_x)
                    ry2 = min(frame_height, y1 + int(person_h * roi_height_ratio))

                    roi = frame[ry1:ry2, rx1:rx2]
                    if roi.size == 0:
                        continue
                    roi_h = roi.shape[0]

                    # Phone detection in ROI
                    roi_results = phone_model.predict(
                        roi,
                        conf=phone_conf,
                        imgsz=640,
                        device=device,
                        half=half,
                        verbose=False,
                    )

                    phone_confirmed = False
                    for det in roi_results[0].boxes:
                        if int(det.cls[0]) != 0:
                            continue
                        bx1, by1, bx2, by2 = map(int, det.xyxy[0])
                        if (by1 + by2) / 2 > roi_h * phone_zone_ratio:
                            continue

                        phone_confirmed = True
                        gx1, gy1 = rx1 + bx1, ry1 + by1
                        gx2, gy2 = rx1 + bx2, ry1 + by2
                        detections.append(Detection(
                            gx1, gy1, gx2, gy2,
                            "phone", float(det.conf[0]),
                            color=COLOR_PHONE,
                        ))
                        break

                    # Initialise state for new tracks
                    if track_id not in usage_votes:
                        usage_votes[track_id]       = deque(maxlen=vote_window)
                        session_start[track_id]     = None
                        display_secs[track_id]      = 0.0
                        prev_stable[track_id]       = False

                    usage_votes[track_id].append(phone_confirmed)
                    positive_votes = sum(usage_votes[track_id])
                    stable_usage   = positive_votes >= vote_required
                    was_stable     = prev_stable[track_id]

                    # Timer logic
                    if stable_usage:
                        last_stable_frame[track_id] = frame_count
                        if not was_stable:
                            session_start[track_id] = frame_count
                            display_secs[track_id]  = 0.0
                            if job_id:
                                self._save_alert(
                                    frame, "phone_usage_confirmed", job_id, frame_count,
                                    extra={"track_id": track_id, "votes": positive_votes},
                                )
                        else:
                            display_secs[track_id] = (frame_count - session_start[track_id]) / fps
                    elif was_stable:
                        session_start[track_id] = None

                    prev_stable[track_id] = stable_usage

                    frames_since_stable = frame_count - last_stable_frame.get(track_id, -hold_frames - 1)
                    show_alert = stable_usage or frames_since_stable <= hold_frames

                    if show_alert:
                        label = f"ID {track_id}  PHONE  {_fmt_time(display_secs[track_id])}"
                        detections.append(Detection(
                            x1, y1, x2, y2, label, 1.0, color=COLOR_PERSON,
                        ))

            status_lines: list[str] = []
            for tid, secs in sorted(display_secs.items()):
                if secs >= 1.0:
                    status_lines.append(f"ID {tid:>3d}: {_fmt_time(secs)}")

            frame_annotations.append(FrameAnnotation(
                frame_idx=frame_count - 1,
                detections=detections,
                status_lines=status_lines,
            ))

        cap.release()

        persons_with_usage = {
            tid: round(secs, 2)
            for tid, secs in display_secs.items()
            if secs > 0
        }

        return KPIResult(
            kpi_name=self.name,
            display_name=self.display_name,
            color=self.color,
            frame_annotations=frame_annotations,
            summary={
                "total_persons_tracked":      len(usage_votes),
                "persons_with_phone_usage":   len(persons_with_usage),
                "usage_seconds_by_person_id": persons_with_usage,
                "total_frames":               frame_count,
                "device":                     device,
            },
        )
