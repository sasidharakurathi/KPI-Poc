# Adding a new KPI

Follow these 5 steps. Nothing outside `app/kpis/` and `config.json` needs to change.

---

## 1. Add your section to `config.json`

Open `config.json` at the project root and add a block keyed by your **class name**:

```json
{
    "YourKPI": {
        "enabled": true,
        "model_path": "path\\to\\your\\model.pt",
        "confidence": 0.5,
        "alert_hold_seconds": 3.0
    }
}
```

| Key | Required | Description |
|-----|----------|-------------|
| `enabled` | no | Set to `false` to disable without removing code (default: `true`) |
| `model_path` | yes | Relative path to your `.pt` weights file |
| anything else | no | Any parameters your KPI needs — read them via `self._get()` |

---

## 2. Create a subfolder

```
app/kpis/
└── your_kpi/
    ├── __init__.py
    └── detector.py
```

---

## 3. Implement `BaseKPI` in `detector.py`

```python
import cv2
from ultralytics import YOLO
from ..base import BaseKPI, Detection, FrameAnnotation, KPIResult
from ..registry import register_kpi
from ...config import settings

# Default values — used when the key is absent from config.json
_DEFAULT_CONF             = 0.5
_DEFAULT_ALERT_HOLD_SECS  = 3.0

@register_kpi
class YourKPI(BaseKPI):
    name = "your_kpi"             # unique snake_case identifier
    display_name = "Your KPI"     # shown in the video overlay panel
    color = (255, 128, 0)         # BGR box colour for this KPI

    def process_video(self, video_path: str) -> KPIResult:
        device = settings.DEVICE
        half   = settings.USE_HALF and device != "cpu"

        # Read every parameter from config.json (second arg is the fallback)
        model_path       = self._get("model_path",         "path/to/default.pt")
        conf             = self._get("confidence",          _DEFAULT_CONF)
        alert_hold_secs  = self._get("alert_hold_seconds", _DEFAULT_ALERT_HOLD_SECS)

        model = YOLO(model_path)
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25

        frame_annotations = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                frame, conf=conf, device=device, half=half, verbose=False
            )
            detections   = []
            status_lines = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = r.names[int(box.cls[0])]
                    conf_val = float(box.conf[0])
                    detections.append(Detection(x1, y1, x2, y2, label, conf_val))

            if detections:
                status_lines.append(f"{len(detections)} detection(s)")

            frame_annotations.append(FrameAnnotation(
                frame_idx=frame_idx,
                detections=detections,
                status_lines=status_lines,
            ))
            frame_idx += 1

        cap.release()

        return KPIResult(
            kpi_name=self.name,
            display_name=self.display_name,
            color=self.color,
            frame_annotations=frame_annotations,
            summary={"total_frames": frame_idx},
        )
```

---

## 4. Export from `__init__.py`

```python
# app/kpis/your_kpi/__init__.py
from .detector import YourKPI  # noqa: F401
```

---

## 5. Register the import in `app/kpis/__init__.py`

Add one line:

```python
from . import your_kpi  # noqa: F401
```

## 6. Adding alerts in a new KPI
Just call:
```python
self._save_alert(
    frame, 
    "your_alert_type", 
    job_id, 
    frame_idx,
    confidence=0.9, 
    extra={"any": "metadata"}
)
```

That's it — the pipeline auto-discovers, configures, and runs your KPI on every uploaded video.

---

## Notes

- **No code change needed for config**: edit `config.json` and call `POST /api/v1/config/reload` — changes take effect on the next job.
- **Disable without removing**: set `"enabled": false` in `config.json`.
- **Stateful models** (tracking, voting windows): keep all state inside `process_video`. Each call gets a fresh instance, so state never leaks between jobs.
- **Per-detection colour**: set `Detection(..., color=(B, G, R))` to override the KPI-level colour for individual boxes.
- **Status panel lines**: anything in `FrameAnnotation.status_lines` appears in the top-left overlay panel next to the KPI's display name.
