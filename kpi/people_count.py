"""
kpi/people_count.py
KPI 1 — Live Count & Total Footfall
Mirrors the core logic from app_original.py, now as a reusable module.
"""
import csv
import cv2
from pathlib import Path
from pydantic import BaseModel
from .base import BaseKPIModule, BaseKPIConfig


class PeopleCountConfig(BaseModel):
    video_source:     str   = "videos/open_crowd.mp4"
    model_path:       str   = "models/yolo26m.pt"
    conf_threshold:   float = 0.35
    min_box_area:     int   = 800
    max_pillar_ratio: float = 4.0
    min_person_ratio: float = 0.6
    delay_ms:         int   = 33
    save_csv:         bool  = True
    csv_path:         str   = "logs/people_count_log.csv"


class PeopleCountKPI(BaseKPIModule):
    """
    Tracks two KPIs:
      • live_count      — people visible in the current frame
      • total_footfall  — unique person IDs seen since session start
    """

    def __init__(self):
        super().__init__()
        self._cfg = PeopleCountConfig()
        self._unique_ids: set = set()
        self._status: dict = {
            "running":        False,
            "frame_number":   0,
            "live_count":     0,
            "total_footfall": 0,
            "fps":            0.0,
        }

    # ── Public interface ──────────────────────────────────────────────────────

    def get_status(self) -> dict:
        return dict(self._status)

    def get_config(self) -> dict:
        return self._cfg.model_dump()

    def set_config(self, data: dict) -> None:
        self._cfg = PeopleCountConfig(**data)

    # ── BaseKPIModule contract ────────────────────────────────────────────────

    def _get_config(self) -> BaseKPIConfig:
        # Bridge Pydantic model → base dataclass-style object
        base = BaseKPIConfig()
        for field in ("video_source", "model_path", "conf_threshold",
                      "min_box_area", "max_pillar_ratio", "min_person_ratio", "delay_ms"):
            setattr(base, field, getattr(self._cfg, field))
        return base

    def _reset_state(self) -> None:
        self._unique_ids.clear()

    def _process_frame(self, frame, valid_detections: list, frame_num: int, fps: float) -> None:
        cfg        = self._cfg
        live_count = len(valid_detections)

        for det in valid_detections:
            self._unique_ids.add(det["track_id"])
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            tid  = det["track_id"]
            conf = det["conf"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{tid} {int(conf*100)}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        total_footfall = len(self._unique_ids)

        # ── CSV log ───────────────────────────────────────────────────────────
        if cfg.save_csv:
            Path(cfg.csv_path).parent.mkdir(parents=True, exist_ok=True)
            write_header = not Path(cfg.csv_path).exists()
            with open(cfg.csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["Frame_Number", "Live_Count", "Total_Footfall"])
                writer.writerow([frame_num, live_count, total_footfall])

        # ── HUD ───────────────────────────────────────────────────────────────
        ov = frame.copy()
        cv2.rectangle(ov, (30, 30), (420, 110), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, f"Live Count    : {live_count}", (40, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Total Footfall: {total_footfall}", (40, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        # ── Status ────────────────────────────────────────────────────────────
        self._status = {
            "running":        True,
            "frame_number":   frame_num,
            "live_count":     live_count,
            "total_footfall": total_footfall,
            "fps":            round(fps, 1),
        }
