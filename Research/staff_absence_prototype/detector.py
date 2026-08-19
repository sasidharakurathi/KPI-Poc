"""People detector with per-camera detection MODE — the "detect every person from any
view" centerpiece of the plan.

No single full-body detector survives every camera angle, so the MODE is chosen per
camera in config.yaml:

    body : full-body COCO person (class 0). Strong on oblique / high-mount / eye-level.
           Works today, cold-start, using the repo's existing yolo26m.pt.
    head : head boxes from a model fine-tuned on CrowdHuman -> person_head.pt.
           The head is the one cue visible from EVERY angle (overhead, crowded,
           occluded), so this is the robust path for top-down / dense cameras.
           Until person_head.pt is trained (Phase 2) the weights_preference falls
           back to the body model so the pipeline still runs.

Optional fisheye dewarp runs BEFORE detection when enabled (Phase 6).

Returns supervision Detections so zones/tracker/annotators compose cleanly.
"""
import cv2
import numpy as np
import torch
import supervision as sv
from ultralytics import YOLO

# Both the COCO person class and a single-class head model use class id 0.
TARGET_CLASS = 0


def _load_first_available(weights_list):
    """Load the first set of weights that succeeds (same idea as pose_backbone)."""
    last_err = None
    for w in weights_list:
        try:
            model = YOLO(w)
            print(f"[detector] loaded weights: {w}")
            return model, w
        except Exception as e:
            last_err = e
            print(f"[detector] could not load {w}: {str(e)[:140]}")
    raise RuntimeError(f"No detector weights could be loaded. Last error: {last_err}")


class FisheyeDewarper:
    """OpenCV fisheye undistort front-end. Off until per-camera intrinsics exist."""
    def __init__(self, fcfg):
        self.K = np.array(fcfg["camera_matrix"], dtype=np.float64)
        self.D = np.array(fcfg["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
        self.balance = float(fcfg.get("balance", 0.0))
        self._maps = None  # built lazily once we know the frame size

    def __call__(self, frame):
        h, w = frame.shape[:2]
        if self._maps is None:
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, self.D, (w, h), np.eye(3), balance=self.balance)
            self._maps = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), newK, (w, h), cv2.CV_16SC2)
        return cv2.remap(frame, self._maps[0], self._maps[1],
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


class PeopleDetector:
    def __init__(self, dcfg):
        self.mode = dcfg.get("mode", "body")
        weights = dcfg["weights_preference"][self.mode]
        self.model, self.weights = _load_first_available(weights)

        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.half = bool(dcfg.get("half", True)) and self.device != "cpu"
        self.conf = float(dcfg.get("conf", 0.35))
        self.iou = float(dcfg.get("iou", 0.50))
        self.imgsz = int(dcfg.get("imgsz", 640))
        self.max_det = int(dcfg.get("max_det", 300))

        self.dewarp = None
        fcfg = dcfg.get("fisheye", {})
        if fcfg.get("enabled", False):
            self.dewarp = FisheyeDewarper(fcfg)

        print(f"[detector] mode={self.mode} device={self.device} half={self.half} "
              f"imgsz={self.imgsz}")

    def preprocess(self, frame):
        """Fisheye dewarp if enabled. Returns the frame detection runs on (also the
        frame zones/overlay must use, since dewarp changes pixel geometry)."""
        return self.dewarp(frame) if self.dewarp is not None else frame

    def detect(self, frame):
        """Run detection on an already-preprocessed frame. Returns sv.Detections
        filtered to people/heads."""
        r = self.model(frame, device=self.device, half=self.half, conf=self.conf,
                       iou=self.iou, imgsz=self.imgsz, max_det=self.max_det,
                       verbose=False)[0]
        det = sv.Detections.from_ultralytics(r)
        return det[det.class_id == TARGET_CLASS]
