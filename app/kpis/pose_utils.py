import numpy as np
from ultralytics import YOLO

from ..config import settings

_DEFAULT_POSE_MODEL_PATH = "app/models/yolo26s-pose.pt"

# COCO-17 shoulder (5,6) and hip (11,12) indices
_KEY_INDICES = [5, 6, 11, 12]


def load_pose_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def run_pose(pose_model: YOLO, frame: np.ndarray):
    return pose_model.predict(
        source=frame,
        device=settings.DEVICE,
        half=settings.USE_HALF and settings.DEVICE != "cpu",
        verbose=False,
    )


def human_keypoints_in_box(pose_results, x1: int, y1: int, x2: int, y2: int) -> bool:
    """
    Returns True if any person in pose_results has shoulder/hip keypoints
    whose centroid (or any individual point) falls inside (x1,y1)-(x2,y2).
    A non-human object produces zero valid keypoints and always returns False.
    """
    for r in pose_results:
        if r.keypoints is None:
            continue
        kps = r.keypoints.xy.cpu().numpy()  # (N, 17, 2)
        for person_kps in kps:
            valid_pts = [
                (float(person_kps[i][0]), float(person_kps[i][1]))
                for i in _KEY_INDICES
                if person_kps[i][0] > 0 and person_kps[i][1] > 0
            ]
            if not valid_pts:
                continue

            cx = sum(p[0] for p in valid_pts) / len(valid_pts)
            cy = sum(p[1] for p in valid_pts) / len(valid_pts)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True

            for kx, ky in valid_pts:
                if x1 <= kx <= x2 and y1 <= ky <= y2:
                    return True

    return False
