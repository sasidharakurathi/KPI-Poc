# Adding a new KPI

Follow these steps. Nothing outside `app/kpis/`, `config.json`, and `app/main.py`'s
`_KPI_LABELS` needs to change.

---

## 1. Add your section to `config.json`

Add a block keyed by your **class name**:

```json
{
    "YourKPI": {
        "enabled": true,
        "model_path": "app/models/your-model.pt",
        "confidence": 0.5
    }
}
```

`enabled` is optional (defaults to `true`). Everything else is whatever
parameters your detector needs - read them back with `self._get(key, default)`.

If the KPI should be assignable to a camera, also add an entry to
`kpi_registry` mapping a numeric catalog ID to your KPI's `name`:

```json
"kpi_registry": { "31": "your_kpi" }
```

---

## 2. Create a subfolder

```
app/kpis/
└── your_kpi/
    ├── __init__.py
    └── detector.py
```

`__init__.py`:
```python
from .detector import YourKPI
```

---

## 3. Implement `BaseKPI` in `detector.py`

```python
import cv2
from ultralytics import YOLO
from ..base import BaseKPI, KPIResult
from ..registry import register_kpi
from ...config import settings


@register_kpi
class YourKPI(BaseKPI):
    name = "your_kpi"
    display_name = "Your KPI"

    def process_video(self, video_path: str, job_id: str = "") -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        model_path = self._get("model_path", "app/models/your-model.pt")
        conf       = self._get("confidence", 0.5)

        model = YOLO(model_path)
        cap   = cv2.VideoCapture(video_path)

        alert_events = 0
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self._observe(frame, frame_idx, job_id)  # every frame, before any skip/stride

            results = model.predict(frame, conf=conf, device=device, half=half, verbose=False)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = results[0].names[int(box.cls[0])]
                alert_events += 1
                self._save_alert(
                    "your_alert_type", job_id, frame_idx,
                    confidence=float(box.conf[0]),
                    extra={"label": label},
                    boxes=[(x1, y1, x2, y2, label, (0, 255, 0))],
                )

            frame_idx += 1

        cap.release()
        self._finalize()

        return KPIResult(self.name, self.display_name, {
            "alert_events": alert_events,
            "total_frames": frame_idx,
            "device": device,
        })
```

---

## 4. Register the import in `app/kpis/__init__.py`

```python
from . import your_kpi
```

---

## Notes

- **Sliding-window frame capture**: call `self._observe(frame, frame_idx, job_id)`
  on every frame you read (even ones you skip via striding), then
  `self._save_alert(...)` on a detection. The base class buffers
  `ALERT_WINDOW_BEFORE` frames before the trigger and waits for
  `ALERT_WINDOW_AFTER` frames after it, then writes the whole raw-frame clip
  to `storage/alerts/{job_id}/{kpi_name}/{alert_id}/` and a row per alert/frame
  to the DB. Call `self._finalize()` once after the read loop to flush any
  window still waiting on trailing frames.
- **`boxes` param on `_save_alert`**: optional. In `DEV_MODE=true`, the anchor
  frame is additionally saved with these boxes drawn on it
  (`labeled_frame{idx}.jpg`), for debugging.
- **Hot-reload config**: edit `config.json` and call `POST /api/config/reload`,
  or use `PUT /api/kpis/{name}/config` to persist changes from the admin UI -
  changes take effect on the next job.
- **Disable without removing**: set `"enabled": false` in `config.json`, or
  `PUT /api/kpis/{name}/config` with `{"enabled": false}`.
- **Stateful models** (tracking, voting windows): keep all state inside
  `process_video` / instance attributes. Each job gets a fresh instance.
- **Zone-based KPIs**: if your detector needs a drawn polygon (a guard post,
  a counting area), set `requires_zone = True` on the class. This makes it
  show up (per-camera) from `GET /api/cameras/{camera_id}/label-frame`, and
  operators can draw + save a polygon via `POST /api/cameras/{camera_id}/labels`
  (`app.services.kpi_label_service`, stored in `KpiZoneLabel`). Read it back
  at runtime with `from ..zone_labels import get_camera_zone_points;
  get_camera_zone_points(job_id, self.name)` - falls back to `None` if this
  camera has no saved polygon yet, so always have a sensible default (e.g.
  the full frame - see `occupancy_dwell`/`staff_absence`/`density_occupancy`).
