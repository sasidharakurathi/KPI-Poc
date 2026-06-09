import cv2
from ultralytics import YOLO
from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ...config import settings

# Default values — used when the key is absent from config.json
_DEFAULT_CONF            = 0.35
_DEFAULT_MIN_BOX_AREA    = 800
_DEFAULT_MAX_PILLAR_RATIO = 4.0
_DEFAULT_MIN_PERSON_RATIO = 0.6


@register_kpi
class PeopleCountKPI(BaseKPI):
    name         = "people_count"        # unique snake_case identifier
    display_name = "People Count"        # shown in the video overlay panel
    color        = (0, 255, 0)           # BGR — green boxes for tracked people

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        # ── Read every parameter from config.json (second arg is the fallback) ──
        model_path        = self._get("model_path",         "yolo26m.pt")
        conf              = self._get("confidence",          _DEFAULT_CONF)
        min_box_area      = self._get("min_box_area",        _DEFAULT_MIN_BOX_AREA)
        max_pillar_ratio  = self._get("max_pillar_ratio",    _DEFAULT_MAX_PILLAR_RATIO)
        min_person_ratio  = self._get("min_person_ratio",    _DEFAULT_MIN_PERSON_RATIO)

        model = YOLO(model_path)

        # Class 0 is always "person" in a single-class people-count model
        TARGET_CLASS_ID = 0

        cap          = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25  # noqa: F841
        frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # noqa: F841
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # noqa: F841

        frame_annotations: list[FrameAnnotation] = []
        frame_count        = 0                    # matches reference naming convention
        unique_people_seen = set()                # cumulative foot-traffic set

        # ── FIX 1: initialise here so summary{} never hits a NameError ──
        total_footfall = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # ── ByteTrack tracking (persist=True keeps IDs consistent) ──
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf,
                device=device,
                half=half,
                verbose=False,
            )

            # ── FIX 2: model.track() returns a list — guard against None/empty ──
            if not results:
                frame_annotations.append(
                    FrameAnnotation(
                        frame_idx=frame_count - 1,
                        detections=[],
                        status_lines=[f"Total foot traffic: {total_footfall}"],
                    )
                )
                continue

            result = results[0]

            detections:   list[Detection] = []
            status_lines: list[str]       = []
            boxes = result.boxes

            # ── FIX 3: guard both boxes AND boxes.id before dereferencing ──
            if boxes is not None and boxes.id is not None:
                track_ids   = boxes.id.int().cpu().tolist()
                cls_indices = boxes.cls.int().cpu().tolist()
                xyxy_list   = boxes.xyxy.int().cpu().tolist()
                confs       = boxes.conf.cpu().tolist()

                for i in range(len(track_ids)):

                    # Filter 1 — target class only
                    if cls_indices[i] != TARGET_CLASS_ID:
                        continue

                    # Filter 2 — confidence threshold
                    if confs[i] < conf:
                        continue

                    x1, y1, x2, y2 = xyxy_list[i]
                    w = x2 - x1
                    h = y2 - y1

                    if w <= 0 or h <= 0:
                        continue

                    area         = w * h
                    aspect_ratio = h / w

                    # Filter 3 — minimum bounding-box area (removes distant noise)
                    if area < min_box_area:
                        continue

                    # Filter 4 — discard impossibly thin/tall pillars
                    if aspect_ratio > max_pillar_ratio:
                        continue

                    # Filter 5 — discard flat/horizontal blobs
                    if aspect_ratio < min_person_ratio:
                        continue

                    # ── Passed all filters ──
                    track_id = track_ids[i]
                    is_new   = track_id not in unique_people_seen
                    unique_people_seen.add(track_id)

                    # ── Save alert snapshot the first time a new person is seen ──
                    if is_new and job_id:
                        self._save_alert(
                            frame,
                            "new_person_detected",
                            job_id,
                            frame_count,
                            extra={"track_id": track_id, "confidence": round(confs[i], 3)},
                            detections=[Detection(x1, y1, x2, y2,
                                                  f"ID {track_id}", confs[i])],
                        )

                    label = (
                        f"ID:{track_id} | "
                        f"{model.names.get(TARGET_CLASS_ID, 'person')} "
                        f"{confs[i]:.2f}"
                    )
                    detections.append(
                        Detection(x1, y1, x2, y2, label, confs[i])
                    )

            live_count = len(detections)
            # ── FIX 4: update total_footfall every frame so summary is always
            #    current regardless of whether boxes.id was None this frame ──
            total_footfall = len(unique_people_seen)

            if live_count > 0:
                status_lines.append(f"Live count: {live_count}")
            status_lines.append(f"Total foot traffic: {total_footfall}")

            frame_annotations.append(
                FrameAnnotation(
                    frame_idx=frame_count - 1,   # 0-based index, matches reference
                    detections=detections,
                    status_lines=status_lines,
                )
            )

        cap.release()

        return KPIResult(
            kpi_name=self.name,
            display_name=self.display_name,
            color=self.color,
            frame_annotations=frame_annotations,
            summary={
                "total_frames":       frame_count,
                "total_foot_traffic": total_footfall,   # always defined — FIX 1
                "device":             device,
            },
        )