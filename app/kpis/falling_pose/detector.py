"""
Falling Pose KPI — pure pose heuristic (no separate falling-pose model).

Uses YOLO pose model to extract keypoints per person, then determines:
  - torso angle vs horizontal
  - bounding-box aspect ratio (wide → horizontal → fallen)

Full-body gate: both sides of shoulders, hips, knees, ankles must be visible
before a fall can be confirmed; gate failures reset the streak.

EventBus debounces per-track streaks with cooldown.
"""
import cv2
import numpy as np

from ... import model_registry
from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ..pose_features import PoseFeatures, L_SH, R_SH, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN
from ..event_bus import EventBus
from ...config import settings

# Shoulder, hip, knee, ankle groups — at least one side must be visible per group
_FULL_BODY_GROUPS = [(L_SH, R_SH), (L_HIP, R_HIP), (L_KN, R_KN), (L_AN, R_AN)]

_BATCH_SIZE = 8


def _full_body_visible(kconf: np.ndarray, thr: float) -> bool:
    for a, b in _FULL_BODY_GROUPS:
        if kconf[a] < thr and kconf[b] < thr:
            return False
    return True


def _is_falling(feat: dict, horiz_angle_thr: float, horiz_aspect_thr: float,
                floor_aspect_thr: float) -> bool:
    angle  = feat.get("torso_angle", 0.0)
    aspect = feat.get("bbox_aspect", 1.0)   # h/w; < 1 means wide/horizontal
    # horizontal torso AND wide bbox
    if angle > horiz_angle_thr and aspect < horiz_aspect_thr:
        return True
    # extremely wide bbox (lying flat) regardless of torso angle
    if aspect < (1.0 / floor_aspect_thr):
        return True
    return False


@register_kpi
class FallingPoseKPI(BaseKPI):
    name = "falling_pose"
    display_name = "Falling Pose"

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        pose_model_path    = self._get("pose_model_path",        "app/models/yolo26m-pose.pt")
        person_conf        = self._get("confidence",              0.30)
        kp_conf            = self._get("kp_conf",                0.30)
        frame_stride       = max(1, self._get("frame_stride",    2))
        require_full_body  = self._get("require_full_body",      True)
        fall_consec        = self._get("fall_consecutive_frames", 5)
        cooldown_secs      = self._get("cooldown_secs",           8.0)
        gap_tol            = self._get("gap_tol",                 2)
        horiz_angle_thr    = self._get("horiz_angle_thr",         60.0)
        horiz_aspect_thr   = self._get("horiz_aspect_thr",        1.1)
        floor_aspect_thr   = self._get("floor_aspect_thr",        1.4)
        infer_imgsz        = self._get("infer_imgsz",             640)

        pose_model = model_registry.get_model(pose_model_path)
        # This model instance may be shared with other KPIs (e.g. mobile_usage,
        # both use yolo26m-pose.pt) or a previous video's job — clear any
        # leftover ByteTrack state before our own persist=True loop starts.
        model_registry.reset_tracker(pose_model)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        bus:    EventBus                  = EventBus(fall_consec, cooldown_secs, gap_tol, fps)
        pf_map: dict[int, PoseFeatures]  = {}

        alert_events = 0
        frame_idx    = 0
        batch: list[tuple[int, np.ndarray]] = []

        def _process_one(fidx: int, r) -> None:
            nonlocal alert_events

            if r.boxes.id is None or r.keypoints is None:
                return

            track_ids  = r.boxes.id.int().cpu().tolist()
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            kps_xy     = r.keypoints.xy.cpu().numpy()
            kps_conf   = r.keypoints.conf.cpu().numpy()

            for tid, bbox, kp, kconf in zip(track_ids, boxes_xyxy, kps_xy, kps_conf):
                # full-body gate
                if require_full_body and not _full_body_visible(kconf, kp_conf):
                    bus.submit(tid, False, fidx)
                    continue

                if tid not in pf_map:
                    pf_map[tid] = PoseFeatures()
                feat = pf_map[tid].update(kp, kconf, tuple(map(int, bbox)), conf_thr=kp_conf)

                falling = _is_falling(feat, horiz_angle_thr, horiz_aspect_thr, floor_aspect_thr)

                if bus.submit(tid, falling, fidx):
                    alert_events += 1
                    x1, y1, x2, y2 = map(int, bbox)
                    self._save_alert(
                        "fall_detected", job_id, fidx,
                        extra={"track_id": int(tid),
                               "torso_angle": round(feat["torso_angle"], 1),
                               "bbox_aspect":  round(feat["bbox_aspect"],  2)},
                        boxes=[(x1, y1, x2, y2, f"ID {tid} FALL", (0, 0, 255))],
                    )

        def _flush_batch() -> None:
            nonlocal batch
            if not batch:
                return
            frames = [f for _, f in batch]
            pose_res = pose_model.track(
                frames, persist=True, tracker="bytetrack.yaml",
                classes=[0], conf=person_conf, imgsz=infer_imgsz,
                device=device, half=half, verbose=False,
            )
            for (fidx, _), r in zip(batch, pose_res):
                _process_one(fidx, r)
            batch = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self._observe(frame, frame_idx, job_id)

            if frame_idx % frame_stride == 0:
                batch.append((frame_idx, frame))
                if len(batch) >= _BATCH_SIZE:
                    _flush_batch()

            frame_idx += 1

        _flush_batch()

        cap.release()
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "alert_events":    alert_events,
            "alarm_triggered": alert_events > 0,
            "total_frames":    frame_idx,
            "device":          device,
        })
