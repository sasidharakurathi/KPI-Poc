"""Falling Pose KPI -- pose heuristic (torso angle + bbox aspect) instead of a separate fall-detection model."""
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
    if angle > horiz_angle_thr and aspect < horiz_aspect_thr:
        return True
    if aspect < (1.0 / floor_aspect_thr):
        return True
    return False


@register_kpi
class FallingPoseKPI(BaseKPI):
    name = "falling_pose"
    display_name = "Falling Pose"

    def setup(self, video_path: str, job_id: str = "") -> None:
        self._job_id = job_id
        self.device = settings.DEVICE
        self.half   = settings.USE_HALF and self.device != "cpu"

        self.pose_model_path    = self._get("pose_model_path",        "app/models/yolo26m-pose.pt")
        self.person_conf        = self._get("confidence",              0.30)
        self.kp_conf            = self._get("kp_conf",                0.30)
        self.frame_stride       = max(1, self._get("frame_stride",    2))
        self.require_full_body  = self._get("require_full_body",      True)
        self.fall_consec        = self._get("fall_consecutive_frames", 5)
        cooldown_secs           = self._get("cooldown_secs",           8.0)
        gap_tol                 = self._get("gap_tol",                 2)
        self.horiz_angle_thr    = self._get("horiz_angle_thr",         60.0)
        self.horiz_aspect_thr   = self._get("horiz_aspect_thr",        1.1)
        self.floor_aspect_thr   = self._get("floor_aspect_thr",        1.4)
        self.infer_imgsz        = self._get("infer_imgsz",             640)

        self.pose_model = model_registry.get_model(self.pose_model_path)
        model_registry.reset_tracker(self.pose_model)   # shared model instance may have stale tracker state

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        self.bus:    EventBus                = EventBus(self.fall_consec, cooldown_secs, gap_tol, fps)
        self.pf_map: dict[int, PoseFeatures] = {}

        self.alert_events = 0
        self._frames_seen = 0
        self.batch: list[tuple[int, np.ndarray]] = []

    def _process_one(self, fidx: int, r) -> None:
        if r.boxes.id is None or r.keypoints is None:
            return

        track_ids  = r.boxes.id.int().cpu().tolist()
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        kps_xy     = r.keypoints.xy.cpu().numpy()
        kps_conf   = r.keypoints.conf.cpu().numpy()

        for tid, bbox, kp, kconf in zip(track_ids, boxes_xyxy, kps_xy, kps_conf):
            if self.require_full_body and not _full_body_visible(kconf, self.kp_conf):
                self.bus.submit(tid, False, fidx)
                continue

            if tid not in self.pf_map:
                self.pf_map[tid] = PoseFeatures()
            feat = self.pf_map[tid].update(kp, kconf, tuple(map(int, bbox)), conf_thr=self.kp_conf)

            falling = _is_falling(feat, self.horiz_angle_thr, self.horiz_aspect_thr, self.floor_aspect_thr)

            if self.bus.submit(tid, falling, fidx):
                self.alert_events += 1
                x1, y1, x2, y2 = map(int, bbox)
                self._save_alert(
                    "fall_detected", self._job_id, fidx,
                    extra={"track_id": int(tid),
                           "torso_angle": round(feat["torso_angle"], 1),
                           "bbox_aspect":  round(feat["bbox_aspect"],  2)},
                    boxes=[(x1, y1, x2, y2, f"ID {tid} FALL", (0, 0, 255))],
                )

    def _flush_batch(self) -> None:
        if not self.batch:
            return
        frames = [f for _, f in self.batch]
        pose_res = self.pose_model.track(
            frames, persist=True, tracker="bytetrack.yaml",
            classes=[0], conf=self.person_conf, imgsz=self.infer_imgsz,
            device=self.device, half=self.half, verbose=False,
        )
        for (fidx, _), r in zip(self.batch, pose_res):
            self._process_one(fidx, r)
        self.batch = []

    def process_frame(self, frame_idx: int, frame: np.ndarray, job_id: str = "") -> None:
        self._observe(frame, frame_idx, self._job_id)

        if frame_idx % self.frame_stride == 0:
            self.batch.append((frame_idx, frame))
            if len(self.batch) >= _BATCH_SIZE:
                self._flush_batch()

        self._frames_seen = frame_idx + 1

    def finalize(self) -> KPIResult:
        self._flush_batch()
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "alert_events":    self.alert_events,
            "alarm_triggered": self.alert_events > 0,
            "total_frames":    self._frames_seen,
            "device":          self.device,
        })
