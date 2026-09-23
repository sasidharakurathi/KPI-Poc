"""ANPR: read license plates of tracked vehicles and log each plate once.

Design choices, each measured on the site footage (1920x1088 overview cameras):
- plates are only accepted inside a tracked vehicle box, since the plate model
  at native resolution also fires on workers, tailgate lettering and rear lights;
- OCR is fast-plate-ocr (cct-s-v2-global): it read the legible plate 50748
  exactly on every crop >= 62px wide, where EasyOCR read "60740";
- plates narrower than `min_plate_width_px` are never read, because below
  ~60px every OCR model returned confident junk ("E11111"), not blanks;
- a plate is confirmed only once `min_votes` reads on the same vehicle agree.
Two-row (square) plates at these distances were unreadable by every model
tested - that needs a camera closer to the lane, not a software change.
"""
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer
from ultralytics import YOLO

from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ...config import settings

VEHICLE_CLASS_IDS = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# UAE-style plate text: optional 1-2 letter code followed by 1-5 digits. Rejects
# words like "NISSAN" that the plate detector boxes on tailgates.
_PLATE_TEXT = re.compile(r"^[A-Z]{0,2}\d{1,5}$")

DEFAULTS = {
    "model_path": "app/models/anpr_lpr.pt",
    "vehicle_model_path": "app/models/yolo26m.pt",
    "ocr_model": "cct-s-v2-global-model",
    "ocr_device": "cpu",
    "confidence": 0.45,
    "vehicle_confidence": 0.25,
    "infer_imgsz": 1920,
    "frame_stride": 2,
    "min_plate_width_px": 60,
    "min_votes": 2,
    "track_buffer": 150,
}


def valid_plate_text(text: str) -> bool:
    return bool(_PLATE_TEXT.match(text))


def _write_tracker_config(track_buffer: int) -> str:
    cfg = (
        "tracker_type: bytetrack\ntrack_high_thresh: 0.25\ntrack_low_thresh: 0.1\n"
        f"new_track_thresh: 0.25\ntrack_buffer: {track_buffer}\nmatch_thresh: 0.8\nfuse_score: True\n"
    )
    fh = tempfile.NamedTemporaryFile("w", suffix="_bytetrack.yaml", delete=False, encoding="utf-8")
    fh.write(cfg)
    fh.close()
    return fh.name


@dataclass
class VehiclePlates:
    votes: Counter = field(default_factory=Counter)
    confirmed: str | None = None
    last_seen: int = 0


@dataclass
class PlateObs:
    box: tuple
    conf: float
    vehicle_id: int | None
    read: str | None          # accepted OCR text this frame, if any
    too_small: bool


@dataclass
class StepResult:
    vehicles: list            # [(tid, class_name, box)]
    plates: list              # [PlateObs]
    confirmed: list           # [(tid, plate_text, votes, plate_box, vehicle_box, class_name)]


class AnprEngine:
    """Per-frame ANPR logic shared by the KPI and the review render script."""

    def __init__(self, cfg: dict, device: str, half: bool):
        self.cfg = {**DEFAULTS, **cfg}
        self.device, self.half = device, half
        self.vehicle_model = YOLO(self.cfg["vehicle_model_path"])
        # Not model_registry: it swaps in anpr_lpr.engine, whose TensorRT profile
        # caps input at 1280px and fails on every 1920px frame (no plates at all).
        self.plate_model = YOLO(self.cfg["model_path"])
        self.ocr = LicensePlateRecognizer(self.cfg["ocr_model"], device=self.cfg["ocr_device"])
        self.ocr_rgb = self.ocr.config.image_color_mode == "rgb"
        self.tracker_cfg = _write_tracker_config(int(self.cfg["track_buffer"]))
        self.tracks: dict[int, VehiclePlates] = {}
        self.seen_plates: set[str] = set()

    def close(self) -> None:
        Path(self.tracker_cfg).unlink(missing_ok=True)

    def _read(self, crop: np.ndarray) -> str:
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB if self.ocr_rgb else cv2.COLOR_BGR2GRAY)
        return self.ocr.run(img)[0].plate.replace("_", "").upper()

    def step(self, frame_idx: int, frame: np.ndarray) -> StepResult:
        c = self.cfg
        vr = self.vehicle_model.track(
            frame, persist=True, tracker=self.tracker_cfg, conf=c["vehicle_confidence"],
            imgsz=c["infer_imgsz"], classes=list(VEHICLE_CLASS_IDS), agnostic_nms=True,
            device=self.device, half=self.half, verbose=False,
        )[0]
        vehicles = []
        if vr.boxes is not None and vr.boxes.id is not None:
            for tid, cls, b in zip(vr.boxes.id.int().tolist(), vr.boxes.cls.int().tolist(),
                                   vr.boxes.xyxy.int().tolist()):
                vehicles.append((tid, VEHICLE_CLASS_IDS.get(cls, "Vehicle"), tuple(b)))
                self.tracks.setdefault(tid, VehiclePlates()).last_seen = frame_idx
        if not vehicles:
            return StepResult([], [], [])

        pr = self.plate_model.predict(frame, conf=c["confidence"], imgsz=c["infer_imgsz"],
                                      device=self.device, half=self.half, verbose=False)[0]
        H, W = frame.shape[:2]
        plates, confirmed = [], []
        for (x1, y1, x2, y2), pconf in zip(pr.boxes.xyxy.int().tolist(), pr.boxes.conf.tolist()):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hosts = [(tid, cls, vb) for tid, cls, vb in vehicles
                     if vb[0] <= cx <= vb[2] and vb[1] <= cy <= vb[3]]
            if not hosts:
                continue   # not on a vehicle - worker, sign, lettering
            tid, cls, vb = min(hosts, key=lambda h: (h[2][2] - h[2][0]) * (h[2][3] - h[2][1]))
            too_small = (x2 - x1) < c["min_plate_width_px"]
            read = None
            if not too_small:
                text = self._read(frame[max(0, y1 - 3):min(H, y2 + 3), max(0, x1 - 3):min(W, x2 + 3)])
                if valid_plate_text(text):
                    read = text
                    track = self.tracks[tid]
                    track.votes[text] += 1
                    if (track.confirmed is None and track.votes[text] >= c["min_votes"]
                            and text not in self.seen_plates):
                        track.confirmed = text
                        self.seen_plates.add(text)
                        confirmed.append((tid, text, track.votes[text], (x1, y1, x2, y2), vb, cls))
            plates.append(PlateObs((x1, y1, x2, y2), pconf, tid, read, too_small))

        # forget vehicles gone for ~10s so a long video doesn't accumulate state
        stale = [t for t, v in self.tracks.items() if frame_idx - v.last_seen > 250]
        for t in stale:
            del self.tracks[t]
        return StepResult(vehicles, plates, confirmed)


@register_kpi
class AnprLprKPI(BaseKPI):
    name         = "ANPR_LPR"
    display_name = "ANPR / License Plate"

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"
        engine = AnprEngine(self._cfg, device, half)
        stride = max(1, int(engine.cfg["frame_stride"]))

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        plates_log: list[dict] = []
        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                self._observe(frame, frame_idx, job_id)
                if frame_idx % stride == 0:
                    for tid, text, votes, pb, vb, cls in engine.step(frame_idx, frame).confirmed:
                        plates_log.append({"plate_text": text, "time_s": round(frame_idx / fps, 1),
                                           "vehicle": cls})
                        self._save_alert(
                            "license_plate_detected", job_id, frame_idx,
                            confidence=1.0,
                            extra={"plate_text": text, "track_id": int(tid), "votes": votes, "vehicle": cls},
                            boxes=[(*vb, f"#{tid} {cls}", (0, 200, 255)), (*pb, text, (0, 200, 0))],
                        )
                frame_idx += 1
        finally:
            cap.release()
            engine.close()
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "alert_events":    len(plates_log),
            "unique_plates":   len(engine.seen_plates),
            "plates_seen":     sorted(engine.seen_plates),
            "plates_log":      plates_log,
            "alarm_triggered": bool(plates_log),
            "total_frames":    frame_idx,
        })
