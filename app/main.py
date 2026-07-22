import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.v1.router import api_router
from .auth.middleware import JWTAuthMiddleware
from .clip_processor import clip_processor
from .config import settings
from .config_loader import get_cameras as get_seed_cameras, get_kpi_config, get_kpi_param, get_kpi_registry
from .db import init_db, seed_cameras, get_config, set_config
from .model_registry import preload_all
from .services import camera_heartbeat
from .services.ws_manager import ws_manager
from .stream_recorder import stream_recorder_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# KPI ID → display label mapping (used by the cameras router)
_KPI_LABELS: dict[int, str] = {
    1: "Unauthorized Access",          2: "ANPR Detection",
    3: "Intrusion Detection",          4: "Vehicle Detection",
    5: "Guard Tour / Hand Detection",  6: "Industrial PPE",
    7: "Fire / Smoke Detection",       8: "Abandoned Object",
    9: "Missing Object",               10: "Intrusion",
    11: "People Count",                12: "Engagement Time / Unattended Guest",
    13: "Staff Absence",               14: "Uniform Detection",
    15: "Carton / Box Detection",      16: "Occupancy",
    17: "Usage Time",                  18: "Fallen Object",
    19: "Guard Presence",              20: "Falling Pose",
    21: "Camera Tampering",            22: "Loading / Unloading Detection",
    23: "Cleaning as per Schedule",    24: "People Density Measurement",
    25: "Opening / Closing Time",      26: "Pack Count",
    27: "Occupancy Count & Dwell Time",28: "Detect Object",
    29: "Smoking",                     30: "Floating Object / Man Overboard",
}


def _enabled_model_paths() -> set[str]:
    from .kpis import get_registry
    paths: set[str] = set()
    for cls in get_registry().values():
        if not get_kpi_param(cls.__name__, "enabled", True):
            continue
        cfg = get_kpi_config(cls.__name__)
        for key, value in cfg.items():
            if "model_path" in key and isinstance(value, str) and value:
                paths.add(value)
    return paths


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    settings.LOGOS_DIR.mkdir(parents=True, exist_ok=True)

    init_db()
    seed_cameras(get_seed_cameras())
    if get_config("timezone") is None:
        set_config("timezone", {"default": "Asia/Riyadh"})

    paths = _enabled_model_paths()
    logger.info(f"[startup] preloading {len(paths)} model(s)…")
    preload_all(paths)

    clip_processor.start()
    stream_recorder_manager.start_all()
    logger.info("[startup] clip processor + IP camera recorders started")

    ws_manager.bind_loop(asyncio.get_running_loop())
    heartbeat_task = asyncio.create_task(camera_heartbeat.run_forever())
    logger.info("[startup] realtime websocket + camera-offline heartbeat monitor started")

    yield

    heartbeat_task.cancel()
    stream_recorder_manager.stop_all()
    clip_processor.stop()


app = FastAPI(
    title="Vision AI Safety & Compliance Platform",
    description=(
        "AI-powered safety and compliance monitoring: KPI detection, camera management, "
        "alerts, and real-time notifications."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(JWTAuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

# Organization logos are non-sensitive brand assets — served as plain static
# files (no auth) rather than through an authenticated endpoint, since a
# login page needs to render an org's logo before anyone has signed in.
# StaticFiles checks the directory exists at mount time, which happens at
# import — before lifespan()'s mkdir has run — so it's created eagerly here.
settings.LOGOS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media/logos", StaticFiles(directory=settings.LOGOS_DIR), name="logos")
