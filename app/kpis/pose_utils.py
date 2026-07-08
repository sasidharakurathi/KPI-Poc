import numpy as np
from ultralytics import YOLO

from .. import model_registry
from ..config import settings

_DEFAULT_POSE_MODEL_PATH = "app/models/yolo26s-pose.pt"

# COCO-17 shoulder (5,6) and hip (11,12) indices
_KEY_INDICES = [5, 6, 11, 12]


def load_pose_model(model_path: str) -> YOLO:
    return model_registry.get_model(model_path)


def run_pose(pose_model: YOLO, frame: np.ndarray, imgsz: int = 640):
    return pose_model.predict(
        source=frame,
        imgsz=imgsz,
        device=settings.DEVICE,
        half=settings.USE_HALF and settings.DEVICE != "cpu",
        verbose=False,
    )


def human_keypoints_in_box(pose_results, x1: int, y1: int, x2: int, y2: int) -> bool:
    """True if any person's shoulder/hip keypoint centroid (or point) falls inside the box."""
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


def _overlap_frac(a, b) -> float:
    """Intersection over the smaller of the two boxes (scale-independent)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / min(area_a, area_b)


def human_confirmed_in_box(
    pose_results, x1: int, y1: int, x2: int, y2: int, box_overlap_thr: float = 0.35
) -> bool:
    """True if the pose model's person box overlaps this box, or shoulder/hip keypoints resolve inside it."""
    target = (x1, y1, x2, y2)
    for r in pose_results:
        boxes = r.boxes
        n = len(boxes) if boxes is not None else 0
        kps = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None

        for i in range(n):
            pbox = tuple(float(v) for v in boxes.xyxy[i].tolist())
            if _overlap_frac(target, pbox) >= box_overlap_thr:
                return True

        if kps is not None:
            for person_kps in kps:
                valid_pts = [
                    (float(person_kps[k][0]), float(person_kps[k][1]))
                    for k in _KEY_INDICES
                    if person_kps[k][0] > 0 and person_kps[k][1] > 0
                ]
                if not valid_pts:
                    continue
                cx = sum(p[0] for p in valid_pts) / len(valid_pts)
                cy = sum(p[1] for p in valid_pts) / len(valid_pts)
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    return True

    return False
