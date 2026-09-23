"""PPE compliance (helmet + vest) per tracked person.

Per sampled frame:
  1. Whole-frame detection gives person boxes; persons are kept only if their
     foot point falls inside the camera's drawn zone (requires_zone - the
     same RestrictArea polygon mechanism other zone KPIs use), else the whole
     frame counts.
  2. ByteTrack assigns persistent IDs, fed at its true sampled rate so the
     lost-track buffer means real seconds.
  3. Crop-refine: every person gets a second detection pass on their own crop
     of the original frame (crop imgsz never below the crop's native size),
     since a whole-frame resize shrinks a vest to too few pixels to classify.
  4. Items found in all crops are pooled in frame coordinates and assigned
     one-to-one to people (_assign_items), so one vest can never make two
     adjacent people compliant.
  5. Displayed status = majority vote over the last few frames per ID, then a
     compliant-hold: once COMPLIANT, stays COMPLIANT until compliant has been
     absent for hold_frames sampled frames.
  6. An alert fires once per ID after alarm_seconds of continuous
     non-compliance.
"""
from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv
from shapely.geometry import Point, Polygon

from ... import model_registry
from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ..zone_labels import get_camera_zone_points
from ...config import settings

PERSON_CLS = "person"
HELMET_CLS = "helmet"
VEST_CLS   = "vest"

_BATCH_SIZE = 4

_STATUS_COLOR = {
    "COMPLIANT": (0, 180, 0),
    "NO VEST":   (0, 140, 255),
    "NO HELMET": (0, 140, 255),
    "NO PPE":    (0, 0, 255),
}


def _box_xyxy(box):
    return [float(x) for x in box]


def _expand_box(box, margin=0.15):
    x1, y1, x2, y2 = box
    w = x2 - x1; h = y2 - y1
    return [x1 - margin*w, y1 - margin*h, x2 + margin*w, y2 + margin*h]


def _box_overlap(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    return (iw*ih) / max(1.0, (b[2]-b[0])*(b[3]-b[1]))


def _raw_status(person_box, helmets, vests, margin, thr):
    expanded = _expand_box(person_box, margin)
    helmet_ok = any(_box_overlap(expanded, h) >= thr for h in helmets)
    vest_ok   = any(_box_overlap(expanded, v) >= thr for v in vests)
    if helmet_ok and vest_ok:   return "COMPLIANT",  (0, 180, 0)
    if helmet_ok:               return "NO VEST",    (0, 140, 255)
    if vest_ok:                 return "NO HELMET",  (0, 140, 255)
    return "NO PPE", (0, 0, 255)


def _assign_items(person_boxes, items, margin, thr):
    """Assigns each detected helmet/vest box to at most one person - whichever
    person's margin-expanded box it overlaps most with. _box_overlap measures
    what fraction of the (small) item box is contained in the (large) person
    box, so without this, a single item straddling two adjacent people can
    independently clear `thr` against both of their boxes and both get
    credited off the same physical PPE item. Returns a list, parallel to
    person_boxes, of the item-lists each person is exclusively assigned.
    """
    expanded = [_expand_box(pb, margin) for pb in person_boxes]
    assigned = [[] for _ in person_boxes]
    for item in items:
        best_i, best_ov = -1, 0.0
        for i, exp in enumerate(expanded):
            ov = _box_overlap(exp, item)
            if ov > best_ov:
                best_ov, best_i = ov, i
        if best_i >= 0 and best_ov >= thr:
            assigned[best_i].append(item)
    return assigned


@register_kpi
class PPEKPI(BaseKPI):
    name = "ppe"
    display_name = "PPE Compliance"
    requires_zone = True

    def setup(self, video_path: str, job_id: str = "") -> None:
        self._job_id = job_id
        self.device = settings.DEVICE
        self.half   = settings.USE_HALF and self.device != "cpu"

        self.model_path    = self._get("model_path",             "app/models/ppe.pt")
        self.conf          = self._get("confidence",             0.25)
        self.margin        = self._get("margin",                 0.15)
        self.overlap_thr   = self._get("overlap_threshold",      0.30)
        alarm_secs         = self._get("alarm_seconds",          2.0)
        sample_fps         = self._get("sample_fps",             12.5)
        self.infer_imgsz   = self._get("infer_imgsz",            960)
        self.refine_margin = self._get("refine_margin",          0.6)
        self.refine_imgsz  = self._get("refine_imgsz",           640)
        self.refine_conf   = self._get("refine_confidence",      0.15)
        self.hold_frames   = self._get("compliant_hold_frames",  10)
        smooth_window      = self._get("smooth_window",          7)

        self.model = model_registry.get_model(self.model_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        # Uploads arrive pre-thinned (~12 fps) while camera recordings are full
        # rate, so a fixed stride would inspect them at different rates. Derive
        # it so both are sampled near sample_fps, the rate the hold/smoothing
        # frame counts were tuned at.
        self.frame_stride = max(1, round(fps / sample_fps))
        sampled_fps = fps / self.frame_stride

        # ByteTrack's lost-track tolerance is lost_track_buffer/30 seconds only
        # when frame_rate is the rate frames actually reach it - here that's the
        # sampled rate, not the source fps or sv's default of 30.
        self.tracker = sv.ByteTrack(
            frame_rate=sampled_fps,
            lost_track_buffer=self._get("track_lost_buffer", 210),
            minimum_matching_threshold=self._get("track_match_threshold", 0.75),
        )
        # counted in sampled frames, since that's what the status loop sees
        self.alarm_frames = max(1, int(alarm_secs * sampled_fps))

        zone_pts = get_camera_zone_points(job_id, self.name) or self._get("zone_points")
        self.zone = Polygon([tuple(p) for p in zone_pts]) if zone_pts and len(zone_pts) >= 3 else None

        self.status_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=smooth_window))
        self.hold_until: dict[int, int] = {}
        self.bad_since: dict[int, int] = {}
        self.alarmed_ids: set[int] = set()

        self.compliant = self.no_helmet = self.no_vest = self.no_ppe = 0
        self.alert_events = 0
        self._frames_seen = 0
        self.batch: list[tuple[int, np.ndarray]] = []

    def _in_zone(self, box) -> bool:
        if self.zone is None:
            return True
        return self.zone.contains(Point((box[0] + box[2]) / 2, box[3]))

    def _crop_items(self, frame: np.ndarray, person_boxes: list) -> tuple[list, list]:
        """Helmet/vest boxes from a detection pass on each person's own crop,
        translated back to frame coordinates and pooled across all crops."""
        fh, fw = frame.shape[:2]
        helmets, vests = [], []
        for x1, y1, x2, y2 in person_boxes:
            bw, bh = x2 - x1, y2 - y1
            cx1 = max(0, int(x1 - self.refine_margin * bw))
            cy1 = max(0, int(y1 - self.refine_margin * bh))
            cx2 = min(fw, int(x2 + self.refine_margin * bw))
            cy2 = min(fh, int(y2 + self.refine_margin * bh))
            if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                continue
            crop = frame[cy1:cy2, cx1:cx2]
            # never below the crop's own resolution - a fixed imgsz would
            # downscale a close person's crop and make them worse, not better
            long_side = max(crop.shape[:2])
            crop_imgsz = max(self.refine_imgsz, ((long_side + 31) // 32) * 32)

            r = self.model.predict(crop, conf=self.refine_conf, imgsz=crop_imgsz,
                                   device=self.device, half=self.half, verbose=False)[0]
            dets = sv.Detections.from_ultralytics(r)
            if not len(dets):
                continue
            for cls_name, out in ((HELMET_CLS, helmets), (VEST_CLS, vests)):
                cls_id = next((k for k, v in r.names.items() if v == cls_name), None)
                if cls_id is None:
                    continue
                for bx1, by1, bx2, by2 in dets[dets.class_id == cls_id].xyxy.tolist():
                    out.append([bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1])
        return helmets, vests

    def _display_status(self, tid: int, raw_status: str, fidx: int) -> str:
        hist = self.status_hist[tid]
        hist.append(raw_status)
        status = max(set(hist), key=list(hist).count)

        if status == "COMPLIANT":
            self.hold_until[tid] = fidx + self.hold_frames * self.frame_stride
        elif fidx <= self.hold_until.get(tid, -1):
            status = "COMPLIANT"
        return status

    def _process_one(self, fidx: int, frame: np.ndarray, persons_sv, person_cls_id) -> None:
        valid_xyxy, valid_conf = [], []
        for i in range(len(persons_sv)):
            box = persons_sv.xyxy[i]
            if not self._in_zone(box):
                continue
            valid_xyxy.append(box)
            valid_conf.append(float(persons_sv.confidence[i]) if persons_sv.confidence is not None else self.conf)

        if not valid_xyxy:
            return

        tracked = self.tracker.update_with_detections(sv.Detections(
            xyxy=np.array(valid_xyxy, dtype=np.float32),
            confidence=np.array(valid_conf, dtype=np.float32),
            class_id=np.full(len(valid_xyxy), person_cls_id, dtype=np.int32),
        ))

        person_boxes = [_box_xyxy(tracked.xyxy[i]) for i in range(len(tracked))]
        helmets, vests = self._crop_items(frame, person_boxes)
        assigned_helmets = _assign_items(person_boxes, helmets, self.margin, self.overlap_thr)
        assigned_vests   = _assign_items(person_boxes, vests,   self.margin, self.overlap_thr)

        for i, pbox in enumerate(person_boxes):
            raw_status, _ = _raw_status(pbox, assigned_helmets[i], assigned_vests[i], self.margin, self.overlap_thr)
            if tracked.tracker_id is None:
                status, tid = raw_status, None
            else:
                tid = int(tracked.tracker_id[i])
                status = self._display_status(tid, raw_status, fidx)

            if status == "COMPLIANT":   self.compliant += 1
            elif status == "NO HELMET": self.no_helmet += 1
            elif status == "NO VEST":   self.no_vest   += 1
            else:                       self.no_ppe    += 1

            if tid is None:
                continue
            if status == "COMPLIANT":
                self.bad_since.pop(tid, None)
                continue

            self.bad_since.setdefault(tid, fidx)
            held = (fidx - self.bad_since[tid]) / self.frame_stride
            if held >= self.alarm_frames and tid not in self.alarmed_ids:
                self.alarmed_ids.add(tid)
                self.alert_events += 1
                pconf = float(tracked.confidence[i]) if tracked.confidence is not None else self.conf
                x1, y1, x2, y2 = map(int, pbox)
                self._save_alert(
                    "ppe_non_compliance", self._job_id, fidx,
                    confidence=round(pconf, 3),
                    extra={"tracker_id": tid, "status": status},
                    boxes=[(x1, y1, x2, y2, f"#{tid} {status}", _STATUS_COLOR[status])],
                )

    def _flush_batch(self) -> None:
        if not self.batch:
            return
        frames = [f for _, f in self.batch]
        results_list = self.model.predict(frames, conf=self.conf, imgsz=self.infer_imgsz,
                                          device=self.device, half=self.half, verbose=False)

        for (fidx, frame), r in zip(self.batch, results_list):
            sv_dets = sv.Detections.from_ultralytics(r)
            person_cls_id = next((k for k, v in r.names.items() if v == PERSON_CLS), None)
            if person_cls_id is None or not len(sv_dets):
                continue
            self._process_one(fidx, frame, sv_dets[sv_dets.class_id == person_cls_id], person_cls_id)

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
            "alert_events":            self.alert_events,
            "compliant_person_frames": self.compliant,
            "no_helmet_person_frames": self.no_helmet,
            "no_vest_person_frames":   self.no_vest,
            "no_ppe_person_frames":    self.no_ppe,
            "alarm_triggered":         len(self.alarmed_ids) > 0,
            "zone_applied":            self.zone is not None,
            "total_frames":            self._frames_seen,
            "device":                  self.device,
        })
