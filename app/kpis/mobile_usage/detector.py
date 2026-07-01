"""
Mobile Phone Usage KPI.

Two-model pipeline:
  1. YOLO pose model → person tracks + keypoints
  2. YOLO phone model → phone box detections

Phone→person association (3-gate):
  a. Bounding-box overlap  ≥ min_overlap_frac
  b. Phone area  ≥ min_area_px  AND aspect ratio within [aspect_min, aspect_max]
  c. Keypoint proximity: phone centre near wrist, head ROI, or body centre

Optional smoking override (batched cigarette model):
  - If a person's upper-body crop contains a cigarette, suppress phone alert for them.

EventBus debounces per-track streaks with cooldown.
"""
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ..pose_features import PoseFeatures
from ..event_bus import EventBus
from ...config import settings


def _iou_frac(box_a, box_b) -> float:
    """Fraction of box_b that overlaps box_a."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / area_b


def _pt_in_box(pt, box) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def _associate_phone(phone_boxes, person_bbox, feat, cfg) -> tuple[bool, list]:
    """Return (phone_detected, matched_phone_boxes)."""
    min_overlap    = cfg["min_overlap_frac"]
    min_area       = cfg["min_area_px"]
    aspect_min     = cfg["aspect_min"]
    aspect_max     = cfg["aspect_max"]
    wrist_thr      = cfg["wrist_threshold"]

    matched = []
    for pb in phone_boxes:
        px1, py1, px2, py2 = pb
        pw = max(1, px2 - px1); ph = max(1, py2 - py1)
        area = pw * ph

        # gate 1: overlap with person box
        if _iou_frac(person_bbox, pb) < min_overlap:
            continue

        # gate 2: size / aspect sanity
        if area < min_area:
            continue
        asp = pw / ph
        if not (aspect_min <= asp <= aspect_max):
            continue

        # gate 3: keypoint proximity
        pcx = (px1 + px2) / 2; pcy = (py1 + py2) / 2
        ph_center = np.array([pcx, pcy])
        scale = feat.get("sh_width", 1.0)

        proximity = False
        # wrist proximity
        for key in ("lwr", "rwr"):
            ok_key = key + "_ok"
            if feat.get(ok_key) and feat.get(key) is not None:
                dist = float(np.linalg.norm(feat[key] - ph_center)) / scale
                if dist < wrist_thr:
                    proximity = True
                    break
        # head ROI proximity
        if not proximity and _pt_in_box(ph_center, feat.get("head_roi", (-1,-1,-1,-1))):
            proximity = True
        # body centre fallback
        if not proximity:
            bc = feat.get("body_center")
            if bc is not None:
                dist = float(np.linalg.norm(bc - ph_center)) / scale
                if dist < wrist_thr * 2:
                    proximity = True

        if proximity:
            matched.append(pb)

    return bool(matched), matched


@register_kpi
class MobileUsageKPI(BaseKPI):
    name = "mobile_usage"
    display_name = "Mobile Phone Usage"

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        pose_model_path = self._get("pose_model_path", "app/models/yolo26m-pose.pt")
        phone_model_path = self._get("phone_model_path", "app/models/yolo26m.pt")
        phone_conf      = self._get("phone_confidence",  0.25)
        kp_conf         = self._get("kp_conf",           0.30)
        person_conf     = self._get("confidence",        0.30)
        frame_stride    = max(1, self._get("frame_stride", 3))
        persist_frames  = self._get("persist_frames",  3)
        cooldown_secs   = self._get("cooldown_secs",   8.0)
        gap_tol         = self._get("gap_tol",         2)

        smoking_override     = self._get("smoking_override", True)
        cig_model_path       = self._get("cigarette_model_path", "app/models/cigarette.pt")
        cig_conf             = self._get("cigarette_confidence", 0.45)
        cig_imgsz            = self._get("cigarette_imgsz", 320)
        upper_body_frac      = self._get("upper_body_fraction", 0.6)

        elbow_angle_max = self._get("elbow_angle_max",  155)
        hand_ear_thr    = self._get("hand_ear_thr",     0.50)

        assoc_cfg = {
            "min_overlap_frac": self._get("min_overlap_frac", 0.15),
            "min_area_px":      self._get("min_area_px",      400),
            "aspect_min":       self._get("aspect_min",       0.20),
            "aspect_max":       self._get("aspect_max",       5.0),
            "wrist_threshold":  self._get("wrist_threshold",  1.5),
        }

        pose_model  = YOLO(pose_model_path)
        phone_model = YOLO(phone_model_path)
        cig_model   = YOLO(cig_model_path) if smoking_override else None

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        bus:        EventBus              = EventBus(persist_frames, cooldown_secs, gap_tol, fps)
        pf_map:     dict[int, PoseFeatures] = {}

        alert_events = 0
        total_tracks: set[int] = set()
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self._observe(frame, frame_idx, job_id)

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            # ── 1. Pose + person tracking ──────────────────────────────────────
            pose_res = pose_model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                classes=[0], conf=person_conf,
                device=device, half=half, verbose=False,
            )
            r = pose_res[0]

            if r.boxes.id is None or r.keypoints is None:
                frame_idx += 1
                continue

            track_ids   = r.boxes.id.int().cpu().tolist()
            boxes_xyxy  = r.boxes.xyxy.cpu().numpy()
            kps_xy      = r.keypoints.xy.cpu().numpy()     # (N, 17, 2)
            kps_conf    = r.keypoints.conf.cpu().numpy()   # (N, 17)
            total_tracks.update(track_ids)

            # ── 2. Cigarette: collect upper-body crops (batched) ──────────────
            crops: list[np.ndarray] = []
            crop_to_person: list[int] = []

            for idx_p, (tid, bbox) in enumerate(zip(track_ids, boxes_xyxy)):
                x1, y1, x2, y2 = map(int, bbox)
                roi_h = int((y2 - y1) * upper_body_frac)
                cy1 = max(0, y1); cy2 = min(fh, y1 + roi_h)
                cx1 = max(0, x1); cx2 = min(fw, x2)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    crops.append(crop)
                    crop_to_person.append(idx_p)

            smoking_persons: set[int] = set()
            if smoking_override and cig_model and crops:
                cig_res = cig_model(
                    crops, conf=cig_conf, imgsz=cig_imgsz,
                    device=device, half=half, verbose=False,
                )
                for j, cr in enumerate(cig_res):
                    if len(cr.boxes) > 0:
                        smoking_persons.add(crop_to_person[j])

            # ── 3. Phone detection on full frame ───────────────────────────────
            phone_res = phone_model.predict(
                frame, conf=phone_conf,
                device=device, half=half, verbose=False,
            )
            phone_boxes_raw = []
            for pr in phone_res:
                for box in pr.boxes:
                    if int(box.cls[0]) == 0:
                        phone_boxes_raw.append(list(map(int, box.xyxy[0])))

            # ── 4. Per-person decision ─────────────────────────────────────────
            smoking_boxes_xyxy = set()
            for sidx in smoking_persons:
                b = boxes_xyxy[sidx]
                smoking_boxes_xyxy.add(tuple(map(int, b)))

            # filter phone boxes that overlap heavily with smoking persons
            filtered_phone_boxes = [
                pb for pb in phone_boxes_raw
                if not any(_iou_frac(tuple(map(int, sb)), pb) > 0.5 for sb in smoking_boxes_xyxy)
            ]

            for idx_p, (tid, bbox, kp, kconf) in enumerate(
                zip(track_ids, boxes_xyxy, kps_xy, kps_conf)
            ):
                if tid not in pf_map:
                    pf_map[tid] = PoseFeatures()

                feat = pf_map[tid].update(kp, kconf, tuple(map(int, bbox)), conf_thr=kp_conf)

                if idx_p in smoking_persons:
                    feat["suppress_mobile"] = True
                    bus.reset(tid)
                    continue

                # associate phone to this person
                phone_hit, matched_phones = _associate_phone(
                    filtered_phone_boxes, tuple(map(int, bbox)), feat, assoc_cfg
                )

                # pose confirmation: elbow bent AND wrist near head
                if phone_hit:
                    elbow_ok = feat["min_elbow_angle"] <= elbow_angle_max
                    wrist_ok = feat["min_wrist_ear"] <= hand_ear_thr or \
                               feat["min_wrist_nose"] <= hand_ear_thr * 1.5
                    phone_hit = elbow_ok or wrist_ok   # either is sufficient

                if bus.submit(tid, phone_hit, frame_idx):
                    alert_events += 1
                    x1, y1, x2, y2 = map(int, bbox)
                    boxes_for_label = [(x1, y1, x2, y2, f"ID {tid} MOBILE", (220, 130, 0))]
                    for mp in matched_phones:
                        boxes_for_label.append((*mp, "phone", (0, 200, 255)))
                    self._save_alert(
                        "phone_usage_confirmed", job_id, frame_idx,
                        extra={"track_id": int(tid)},
                        boxes=boxes_for_label,
                    )

            frame_idx += 1

        cap.release()
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "alert_events":        alert_events,
            "total_persons_tracked": len(total_tracks),
            "total_frames":        frame_idx,
            "device":              device,
        })
