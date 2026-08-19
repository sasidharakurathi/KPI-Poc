"""Mobile Phone Usage KPI -- pose model tracks persons, phone model detects phones, 3-gate association links them; a cigarette-crop override suppresses false positives from smoking."""
import cv2
import numpy as np
from collections import defaultdict

from ... import model_registry
from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ..pose_features import PoseFeatures
from ..event_bus import EventBus
from ...config import settings

_BATCH_SIZE = 4


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
    max_area_frac  = cfg["max_area_frac_of_person"]

    px1p, py1p, px2p, py2p = person_bbox
    person_area = max(1, (px2p - px1p) * (py2p - py1p))

    matched = []
    for pb in phone_boxes:
        px1, py1, px2, py2 = pb
        pw = max(1, px2 - px1); ph = max(1, py2 - py1)
        area = pw * ph

        # gate 1: overlap with person box
        if _iou_frac(person_bbox, pb) < min_overlap:
            continue

        # gate 2: size / aspect sanity
        if area < min_area or area > max_area_frac * person_area:
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

    def setup(self, video_path: str, job_id: str = "") -> None:
        self._job_id = job_id
        self.device = settings.DEVICE
        self.half   = settings.USE_HALF and self.device != "cpu"

        self.pose_model_path  = self._get("pose_model_path", "app/models/yolo26m-pose.pt")
        self.phone_model_path = self._get("phone_model_path", "app/models/yolo26m.pt")
        self.phone_conf       = self._get("phone_confidence",  0.25)
        self.kp_conf          = self._get("kp_conf",           0.30)
        self.person_conf      = self._get("confidence",        0.30)
        self.frame_stride     = max(1, self._get("frame_stride", 3))
        self.pose_imgsz       = self._get("pose_imgsz",  640)
        self.phone_imgsz      = self._get("phone_imgsz", 640)
        persist_frames        = self._get("persist_frames",  3)
        cooldown_secs         = self._get("cooldown_secs",   8.0)
        gap_tol                = self._get("gap_tol",         2)

        self.smoking_override = self._get("smoking_override", True)
        self.cig_model_path   = self._get("cigarette_model_path", "app/models/cigarette.pt")
        self.cig_conf         = self._get("cigarette_confidence", 0.45)
        self.cig_imgsz        = self._get("cigarette_imgsz", 320)
        self.upper_body_frac  = self._get("upper_body_fraction", 0.6)

        self.elbow_angle_max = self._get("elbow_angle_max",  155)
        self.hand_ear_thr    = self._get("hand_ear_thr",     0.50)

        self.assoc_cfg = {
            "min_overlap_frac": self._get("min_overlap_frac", 0.15),
            "min_area_px":      self._get("min_area_px",      400),
            "aspect_min":       self._get("aspect_min",       0.20),
            "aspect_max":       self._get("aspect_max",       5.0),
            "wrist_threshold":  self._get("wrist_threshold",  1.5),
            "max_area_frac_of_person": self._get("max_area_frac_of_person", 0.12),
        }

        self.pose_model  = model_registry.get_model(self.pose_model_path)
        model_registry.reset_tracker(self.pose_model)   # shared model instance may have stale tracker state
        self.cig_model = model_registry.get_model(self.cig_model_path) if self.smoking_override else None

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self.bus:    EventBus                = EventBus(persist_frames, cooldown_secs, gap_tol, fps)
        self.pf_map: dict[int, PoseFeatures] = {}

        self.alert_events = 0
        self.total_tracks: set[int] = set()
        self._frames_seen = 0
        self.batch: list[tuple[int, np.ndarray]] = []

    def _process_one(self, fidx: int, frame: np.ndarray, r, phone_boxes_raw_for_frame: list) -> None:
        if r.boxes.id is None or r.keypoints is None:
            return

        track_ids   = r.boxes.id.int().cpu().tolist()
        boxes_xyxy  = r.boxes.xyxy.cpu().numpy()
        kps_xy      = r.keypoints.xy.cpu().numpy()     # (N, 17, 2)
        kps_conf    = r.keypoints.conf.cpu().numpy()   # (N, 17)
        self.total_tracks.update(track_ids)

        feats: list[dict] = []
        for tid, bbox, kp, kconf in zip(track_ids, boxes_xyxy, kps_xy, kps_conf):
            if tid not in self.pf_map:
                self.pf_map[tid] = PoseFeatures()
            feats.append(self.pf_map[tid].update(kp, kconf, tuple(map(int, bbox)), conf_thr=self.kp_conf))

        # collect upper-body + wrist-region crops for the cigarette model
        crops: list[np.ndarray] = []
        crop_to_person: list[int] = []

        for idx_p, (tid, bbox, feat) in enumerate(zip(track_ids, boxes_xyxy, feats)):
            x1, y1, x2, y2 = map(int, bbox)

            roi_h = int((y2 - y1) * self.upper_body_frac)
            cy1 = max(0, y1); cy2 = min(self.fh, y1 + roi_h)
            cx1 = max(0, x1); cx2 = min(self.fw, x2)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                crops.append(crop)
                crop_to_person.append(idx_p)

            scale = feat.get("sh_width") or (x2 - x1) * 0.30
            half_span = max(40, int(scale * 1.2))
            for key, ok_key in (("lwr", "lwr_ok"), ("rwr", "rwr_ok")):
                if not feat.get(ok_key):
                    continue
                wx, wy = feat[key]
                wcx1 = max(0, int(wx - half_span)); wcy1 = max(0, int(wy - half_span))
                wcx2 = min(self.fw, int(wx + half_span)); wcy2 = min(self.fh, int(wy + half_span))
                wcrop = frame[wcy1:wcy2, wcx1:wcx2]
                if wcrop.size > 0:
                    crops.append(wcrop)
                    crop_to_person.append(idx_p)

        smoking_persons: set[int] = set()
        if self.smoking_override and self.cig_model and crops:
            cig_res = self.cig_model(
                crops, conf=self.cig_conf, imgsz=self.cig_imgsz,
                device=self.device, half=self.half, verbose=False,
            )
            for j, cr in enumerate(cig_res):
                if len(cr.boxes) > 0:
                    smoking_persons.add(crop_to_person[j])

        phone_boxes_raw = []
        phone_conf_by_box: dict[tuple, float] = {}
        for x1, y1, x2, y2, cls_id, conf in phone_boxes_raw_for_frame:
            if cls_id == 67 and conf >= self.phone_conf:   # yolo26m.pt (COCO): 67 = cell phone
                b = [int(x1), int(y1), int(x2), int(y2)]
                phone_boxes_raw.append(b)
                phone_conf_by_box[tuple(b)] = conf

        smoking_boxes_xyxy = set()
        for sidx in smoking_persons:
            b = boxes_xyxy[sidx]
            smoking_boxes_xyxy.add(tuple(map(int, b)))

        filtered_phone_boxes = [
            pb for pb in phone_boxes_raw
            if not any(_iou_frac(tuple(map(int, sb)), pb) > 0.5 for sb in smoking_boxes_xyxy)
        ]

        for idx_p, (tid, bbox, feat) in enumerate(
            zip(track_ids, boxes_xyxy, feats)
        ):
            if idx_p in smoking_persons:
                feat["suppress_mobile"] = True
                self.bus.submit(tid, False, fidx)
                continue

            phone_hit, matched_phones = _associate_phone(
                filtered_phone_boxes, tuple(map(int, bbox)), feat, self.assoc_cfg
            )

            if phone_hit:
                elbow_ok = feat["min_elbow_angle"] <= self.elbow_angle_max
                wrist_ok = feat["min_wrist_ear"] <= self.hand_ear_thr or \
                           feat["min_wrist_nose"] <= self.hand_ear_thr * 1.5
                phone_hit = elbow_ok or wrist_ok   # either is sufficient

            if self.bus.submit(tid, phone_hit, fidx):
                self.alert_events += 1
                x1, y1, x2, y2 = map(int, bbox)
                boxes_for_label = [(x1, y1, x2, y2, f"ID {tid} MOBILE", (220, 130, 0))]
                for mp in matched_phones:
                    boxes_for_label.append((*mp, "phone", (0, 200, 255)))
                best_conf = max(
                    (phone_conf_by_box.get(tuple(mp), self.phone_conf) for mp in matched_phones),
                    default=self.phone_conf,
                )
                self._save_alert(
                    "phone_usage_confirmed", self._job_id, fidx,
                    confidence=round(best_conf, 3),
                    extra={"track_id": int(tid)},
                    boxes=boxes_for_label,
                )

    def _flush_batch(self) -> None:
        if not self.batch:
            return
        frames = [f for _, f in self.batch]

        pose_res_list = self.pose_model.track(
            frames, persist=True, tracker="bytetrack.yaml",
            classes=[0], conf=self.person_conf, imgsz=self.pose_imgsz,
            device=self.device, half=self.half, verbose=False,
        )

        phone_boxes_by_frame = self.shared_cache.predict_boxes_batch(
            self.phone_model_path, self.batch, self.phone_imgsz, self.device, self.half
        )

        for (fidx, frame), r in zip(self.batch, pose_res_list):
            self._process_one(fidx, frame, r, phone_boxes_by_frame.get(fidx, []))

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
            "alert_events":        self.alert_events,
            "total_persons_tracked": len(self.total_tracks),
            "alarm_triggered":     self.alert_events > 0,
            "total_frames":        self._frames_seen,
            "device":              self.device,
        })
