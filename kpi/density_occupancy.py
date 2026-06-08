"""
kpi/density_occupancy.py
KPI 2 — Density & Occupancy
Pure math on top of live_count — no additional model needed.
  density        = live_count / zone_sqft
  occupancy_alert = live_count > max_occupancy
"""
import csv
import cv2
from pathlib import Path
from pydantic import BaseModel
from .base import BaseKPIModule, BaseKPIConfig


class DensityOccupancyConfig(BaseModel):
    video_source:     str   = "videos/open_crowd.mp4"
    model_path:       str   = "models/yolo26m.pt"
    conf_threshold:   float = 0.35
    min_box_area:     int   = 800
    max_pillar_ratio: float = 4.0
    min_person_ratio: float = 0.6
    delay_ms:         int   = 33
    # Density / occupancy parameters
    zone_sqft:        float = 500.0
    max_occupancy:    int   = 50
    density_low:      float = 0.02   # ppl/sqft thresholds
    density_medium:   float = 0.05
    density_high:     float = 0.10
    save_csv:         bool  = True
    csv_path:         str   = "logs/density_occupancy_log.csv"


_LEVEL_COLORS = {
    "LOW":      (0, 200, 80),
    "MEDIUM":   (0, 200, 255),
    "HIGH":     (0, 130, 255),
    "CRITICAL": (0, 0, 255),
    "NONE":     (170, 170, 170),
}


class DensityOccupancyKPI(BaseKPIModule):
    """
    Adds density and occupancy analytics on top of the person detector.
    Density levels:
        LOW      → density < density_low
        MEDIUM   → density_low  ≤ density < density_medium
        HIGH     → density_medium ≤ density < density_high
        CRITICAL → density ≥ density_high
    """

    def __init__(self):
        super().__init__()
        self._cfg = DensityOccupancyConfig()
        self._status: dict = {
            "running":         False,
            "frame_number":    0,
            "live_count":      0,
            "density":         0.0,
            "density_level":   "NONE",
            "occupancy_alert": False,
            "max_occupancy":   50,
            "zone_sqft":       500.0,
            "fps":             0.0,
        }

    # ── Public interface ──────────────────────────────────────────────────────

    def get_status(self) -> dict:
        return dict(self._status)

    def get_config(self) -> dict:
        return self._cfg.model_dump()

    def set_config(self, data: dict) -> None:
        self._cfg = DensityOccupancyConfig(**data)

    # ── BaseKPIModule contract ────────────────────────────────────────────────

    def _get_config(self) -> BaseKPIConfig:
        base = BaseKPIConfig()
        for field in ("video_source", "model_path", "conf_threshold",
                      "min_box_area", "max_pillar_ratio", "min_person_ratio", "delay_ms"):
            setattr(base, field, getattr(self._cfg, field))
        return base

    def _reset_state(self) -> None:
        pass  # Density is stateless — recalculated fresh each frame

    def _process_frame(self, frame, valid_detections: list, frame_num: int, fps: float) -> None:
        cfg        = self._cfg
        live_count = len(valid_detections)

        # Draw bounding boxes
        for det in valid_detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ── Analytics math ────────────────────────────────────────────────────
        density       = live_count / cfg.zone_sqft if cfg.zone_sqft > 0 else 0.0
        density_level = self._level(density)
        alert         = live_count > cfg.max_occupancy

        # ── CSV log ───────────────────────────────────────────────────────────
        if cfg.save_csv:
            Path(cfg.csv_path).parent.mkdir(parents=True, exist_ok=True)
            write_header = not Path(cfg.csv_path).exists()
            with open(cfg.csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "Frame_Number", "Live_Count", "Density_ppsf",
                        "Density_Level", "Occupancy_Alert"
                    ])
                writer.writerow([
                    frame_num, live_count,
                    round(density, 5), density_level, int(alert)
                ])

        # ── HUD ───────────────────────────────────────────────────────────────
        d_col   = _LEVEL_COLORS.get(density_level, (170, 170, 170))
        occ_col = (0, 0, 255) if alert else (0, 255, 100)
        rows = [
            (f"Live Count      : {live_count}",                            (0, 0, 255)),
            (f"Density         : {density:.4f} ppl/sqft [{density_level}]", d_col),
            (f"Occupancy Alert : {'YES ⚠' if alert else 'NO'}",            occ_col),
            (f"FPS             : {fps:.1f}",                               (170, 170, 170)),
        ]
        ov = frame.copy()
        cv2.rectangle(ov, (30, 30), (530, 40 + len(rows) * 32), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        for i, (txt, col) in enumerate(rows):
            cv2.putText(frame, txt, (40, 58 + i * 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

        # ── Status ────────────────────────────────────────────────────────────
        self._status = {
            "running":         True,
            "frame_number":    frame_num,
            "live_count":      live_count,
            "density":         round(density, 5),
            "density_level":   density_level,
            "occupancy_alert": alert,
            "max_occupancy":   cfg.max_occupancy,
            "zone_sqft":       cfg.zone_sqft,
            "fps":             round(fps, 1),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _level(self, density: float) -> str:
        cfg = self._cfg
        if density < cfg.density_low:    return "LOW"
        if density < cfg.density_medium: return "MEDIUM"
        if density < cfg.density_high:   return "HIGH"
        return "CRITICAL"
