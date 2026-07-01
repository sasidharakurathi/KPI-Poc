# KPI Video Analytics

A computer-vision platform that runs pluggable KPI detectors (PPE compliance, fire/smoke,
mobile phone usage, falling pose, smoking, ANPR, people counting, object detection,
density/occupancy, floating object, vehicle speed) over uploaded video, and lets an admin
configure cameras and detection parameters through a web UI.

The system has two parts:

- **`app/`** — FastAPI backend: video pipeline, KPI plugin system, Postgres-backed job/alert
  storage, and the admin/config REST API.
- **`vision-ai-frontend/`** — Angular 18 frontend: camera dashboard, admin pages (cameras, KPI
  settings), and an alert browser with per-detection frame evidence.

---

## How it works

1. An admin registers a **camera** (a position/name/zone/priority) and assigns it a set of
   KPIs to run.
2. A video is uploaded for that camera (`POST /api/videos/upload`). The backend resolves the
   camera's assigned KPIs and runs each one, in parallel, over the video.
3. Each KPI detector reads the video frame by frame. On a detection, it saves an **alert**: a
   sliding window of raw frames (`ALERT_WINDOW_BEFORE` frames + the trigger frame +
   `ALERT_WINDOW_AFTER` frames) to disk, plus a DB row. In developer mode, it also saves one
   labeled copy of the trigger frame with detection boxes drawn on it.
4. The frontend polls job status, then lets you browse every alert with its frame evidence.

There is no live/RTSP camera ingestion — detection runs against uploaded video files only.

---

## Backend (`app/`)

| File | Responsibility |
|---|---|
| `main.py` | FastAPI routes — cameras, KPI settings, video upload/status, alerts |
| `pipeline.py` | Runs a job's assigned KPIs concurrently in a thread pool |
| `db.py` | SQLModel tables (`Job`, `Alert`, `AlertFrame`, `Camera`) + Postgres access |
| `job_manager.py` | Thin wrapper over `db.py` for job create/read/update |
| `config.py` | `.env`-driven runtime settings (device, paths, alert window size, ...) |
| `config_loader.py` | Reads/writes `config.json` — per-KPI parameters and the KPI-ID registry |
| `kpi_logger.py` | Per-model resource usage (CPU/RAM/GPU) logging per pipeline run |
| `schemas.py` | Pydantic request/response models |
| `kpis/base.py` | `BaseKPI` — the sliding-window alert capture every detector inherits |
| `kpis/registry.py` | `@register_kpi` class decorator + registry lookup |
| `kpis/pose_features.py`, `kpis/pose_utils.py`, `kpis/event_bus.py` | Shared helpers (pose keypoint math, debounce/cooldown logic) used by multiple detectors |
| `kpis/<name>/detector.py` | One folder per KPI — see `kpis/ADDING_A_KPI.md` to add a new one |

### KPI catalog vs. real detectors

`config.json` defines two related but distinct things:

- **`kpi_registry`**: maps a numeric catalog ID (1-30, a fixed reference list covering both
  implemented and future-roadmap KPIs) to the snake_case `name` of a real, registered
  detector. Only IDs with an entry here are functional; the rest exist so a camera can be
  pre-configured for a KPI before it's built.
- **`cameras`**: seed data for the `Camera` DB table (see below) — each camera lists which
  numeric KPI IDs it wants. Only the ones with a `kpi_registry` mapping actually run.

Eleven KPIs are currently implemented: `mobile_usage`, `falling_pose`, `smoking`, `ppe`,
`fire_smoke`, `people_count`, `object_detection`, `ANPR_LPR`, `floating`,
`density_occupancy`, `speed_tracker`.

### Data model

- **`Camera`** — `camera_id` (PK), `name`, `zone`, `priority`, `kpi_ids` (list of catalog IDs).
  Lives in Postgres; seeded once from `config.json["cameras"]` on first startup, then fully
  managed via the admin API/UI (`config.json`'s copy is never written back to).
- **`Job`** — one row per uploaded video: status, which KPIs ran, and their result summaries.
- **`Alert`** — one row per detection: which KPI/job it belongs to, confidence, and a
  free-form `extra` dict for detector-specific metadata (track IDs, angles, etc.).
- **`AlertFrame`** — one row per saved frame in an alert's clip window, with its disk path.

### Configuring a KPI's detection parameters

Every KPI reads its tunable parameters via `self._get(key, default)`, backed by
`config.json[ClassName]`. Change them either by editing `config.json` directly and calling
`POST /api/config/reload`, or through the admin UI (`PUT /api/kpis/{name}/config`), which
persists to the same file. Parameters are global per KPI — not per camera.

### Running the backend

```bash
pip install -r requirements.txt
cp .env.example .env   # set DATABASE_URL to a running Postgres instance
uvicorn app.main:app --reload
```

Key `.env` settings: `DATABASE_URL`, `DEVICE` (`cuda:0` / `cpu`), `USE_HALF`, `DEV_MODE`
(saves labeled evidence frames), `ALERT_WINDOW_BEFORE` / `ALERT_WINDOW_AFTER`, `MAX_WORKERS`.

### API surface

| Area | Routes |
|---|---|
| Cameras | `GET/POST /api/cameras`, `GET/PUT/DELETE /api/cameras/{id}` |
| KPI catalog | `GET /api/kpis` (enabled only), `GET /api/kpis/settings` (all, with config), `PUT /api/kpis/{name}/config` |
| Video jobs | `POST /api/videos/upload`, `GET /api/videos/{job_id}/status`, `GET /api/videos/{job_id}/alerts` |
| Alerts | `GET /api/alerts` (filter by `camera_id`/`kpi_name`/`job_id`), `GET /api/alerts/{id}`, `GET /api/alerts/{id}/frames/{position}`, `GET /api/alerts/{id}/labeled` |
| Config | `GET /api/config`, `POST /api/config/reload` |

---

## Frontend (`vision-ai-frontend/`)

Angular 18, standalone components, signals for state, plain CSS (no UI framework), native
Fetch-based `HttpClient`.

```
src/app/
├── pages/            # routed pages
│   ├── dashboard/         camera overview + video upload (/dashboard)
│   ├── cameras-admin/     camera CRUD + KPI assignment (/admin/cameras)
│   ├── kpi-settings/      per-KPI enable + parameter forms (/admin/kpis)
│   ├── alerts-list/       filterable alert grid (/alerts)
│   └── alert-detail/      8-frame filmstrip + metadata (/alerts/:id)
├── components/       # dashboard building blocks (topbar, camera rail, upload panel, ...)
├── services/         # one per backend resource: camera, kpi, alerts, video, toast
├── models/           # TypeScript interfaces mirroring the FastAPI schemas
└── data/             # static KPI catalog labels (id → display name)
```

`PortVisionService` is the shared app state (selected camera, camera list, live alert feed).
Each page/service otherwise talks to the backend directly through its own service.

### Running the frontend

```bash
cd vision-ai-frontend
npm install
npm start   # http://localhost:4200, expects the API at http://127.0.0.1:8000/api
```

The API base URL is set in `src/environments/environment.ts`.
