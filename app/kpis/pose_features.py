"""Per-track pose feature extractor -- turns YOLO pose keypoints into a feature dict for MobileUsageKPI and FallingPoseKPI."""
from collections import deque
from typing import Optional

import numpy as np

NOSE  = 0
L_EAR = 3;  R_EAR = 4
L_SH  = 5;  R_SH  = 6
L_EL  = 7;  R_EL  = 8
L_WR  = 9;  R_WR  = 10
L_HIP = 11; R_HIP = 12
L_KN  = 13; R_KN  = 14
L_AN  = 15; R_AN  = 16


def _angle_at_joint(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at joint b (degrees)."""
    v1 = a - b; v2 = c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))))


class PoseFeatures:
    """Stateful, per-track feature extractor with a ring buffer for temporal cues."""

    def __init__(self, ring_size: int = 48):
        self._ring: deque = deque(maxlen=ring_size)

    def reset(self) -> None:
        self._ring.clear()

    def update(
        self,
        kp: np.ndarray,       # (17, 2) keypoint x,y
        kconf: np.ndarray,    # (17,)   keypoint confidences
        bbox: tuple,          # (x1, y1, x2, y2)
        conf_thr: float = 0.30,
    ) -> dict:
        """Return a feature dict for this frame."""
        feat: dict = {}

        ok = kconf >= conf_thr   # bool mask, shape (17,)

        bw = max(1.0, float(bbox[2] - bbox[0]))
        bh = max(1.0, float(bbox[3] - bbox[1]))

        if ok[L_SH] and ok[R_SH]:
            sh_width = max(1.0, float(np.linalg.norm(kp[L_SH] - kp[R_SH])))
        else:
            sh_width = max(1.0, bw * 0.30)
        feat["sh_width"] = sh_width
        scale = sh_width

        head_pts = [kp[i] for i in (NOSE, L_EAR, R_EAR) if ok[i]]
        if head_pts:
            head_center = np.mean(head_pts, axis=0)
        else:
            head_center = np.array([(bbox[0]+bbox[2])/2,
                                    bbox[1] + bh * 0.10])
        feat["head_center"] = head_center
        hs = sh_width * 0.70
        feat["head_roi"] = (
            int(head_center[0] - hs), int(head_center[1] - hs),
            int(head_center[0] + hs), int(head_center[1] + hs),
        )
        feat["head_area"] = max(1.0, (2 * hs) ** 2)

        feat["lwr"] = kp[L_WR]; feat["rwr"] = kp[R_WR]
        feat["lwr_ok"] = bool(ok[L_WR]); feat["rwr_ok"] = bool(ok[R_WR])

        we_dists = []
        for wi in (L_WR, R_WR):
            if not ok[wi]: continue
            for ei in (L_EAR, R_EAR):
                if ok[ei]:
                    we_dists.append(np.linalg.norm(kp[wi] - kp[ei]) / scale)
        feat["min_wrist_ear"] = float(min(we_dists)) if we_dists else 99.0

        wn_dists = []
        if ok[NOSE]:
            for wi in (L_WR, R_WR):
                if ok[wi]:
                    wn_dists.append(np.linalg.norm(kp[wi] - kp[NOSE]) / scale)
        feat["min_wrist_nose"] = float(min(wn_dists)) if wn_dists else 99.0

        elbow_angles = []
        if ok[L_SH] and ok[L_EL] and ok[L_WR]:
            elbow_angles.append(_angle_at_joint(kp[L_SH], kp[L_EL], kp[L_WR]))
        if ok[R_SH] and ok[R_EL] and ok[R_WR]:
            elbow_angles.append(_angle_at_joint(kp[R_SH], kp[R_EL], kp[R_WR]))
        feat["min_elbow_angle"] = float(min(elbow_angles)) if elbow_angles else 180.0

        if ok[L_SH] and ok[R_SH]:
            dy = float(kp[R_SH][1] - kp[L_SH][1])
            dx = float(kp[R_SH][0] - kp[L_SH][0]) + 1e-6
            feat["torso_angle"] = abs(float(np.degrees(np.arctan2(dy, dx))))
        else:
            feat["torso_angle"] = 0.0

        feat["bbox_aspect"] = bh / bw

        feat["body_center"] = np.array([(bbox[0]+bbox[2])/2,
                                        (bbox[1]+bbox[3])/2])

        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self._ring.append(np.array([cx, cy]))

        if len(self._ring) >= 3:
            recent = list(self._ring)
            vel = float(np.linalg.norm(recent[-1] - recent[0])) / (scale + 1e-6)
            feat["velocity"] = vel / max(1, len(recent) - 1)
        else:
            feat["velocity"] = 0.0

        feat["suppress_mobile"] = False   # overridden externally if needed
        return feat
