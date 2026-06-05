import json
import threading
from pathlib import Path
from typing import Any, Optional

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

_cache: dict = {}
_lock = threading.Lock()


def _load_from_disk() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _ensure_loaded() -> None:
    global _cache
    if not _cache:
        with _lock:
            if not _cache:
                _cache = _load_from_disk()


def reload() -> dict:
    """Re-read config.json from disk. Returns the new config."""
    global _cache
    with _lock:
        _cache = _load_from_disk()
    return _cache


def get_all() -> dict:
    _ensure_loaded()
    return dict(_cache)


# ── KPI class config ─────────────────────────────────────────────────────────

def get_kpi_config(kpi_class_name: str) -> dict:
    """Return the config block for a KPI class, or {} if not present."""
    _ensure_loaded()
    return dict(_cache.get(kpi_class_name, {}))


def get_kpi_param(kpi_class_name: str, key: str, default: Any = None) -> Any:
    return get_kpi_config(kpi_class_name).get(key, default)


# ── KPI registry (numeric ID → KPI name) ─────────────────────────────────────

def get_kpi_registry() -> dict[str, str]:
    """
    Returns the kpi_registry mapping: {str(kpi_id): kpi_name}.
    e.g. {"7": "fire_smoke", "17": "mobile_usage"}
    """
    _ensure_loaded()
    return dict(_cache.get("kpi_registry", {}))


def resolve_kpi_names(kpi_ids: list[int]) -> list[str]:
    """
    Convert a list of numeric KPI IDs to implemented KPI names.
    IDs without a mapping in kpi_registry are silently skipped.
    """
    registry = get_kpi_registry()
    seen, names = set(), []
    for kid in kpi_ids:
        name = registry.get(str(kid))
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


# ── Camera registry ───────────────────────────────────────────────────────────

def get_cameras() -> dict[str, dict]:
    """Return the full cameras dict keyed by camera ID."""
    _ensure_loaded()
    return dict(_cache.get("cameras", {}))


def get_camera(camera_id: str) -> Optional[dict]:
    """Return a single camera's config or None if not found."""
    return get_cameras().get(camera_id)


def get_camera_kpi_names(camera_id: str) -> list[str]:
    """
    Return the list of implemented KPI names for a given camera.
    Unimplemented KPI IDs are skipped automatically.
    """
    cam = get_camera(camera_id)
    if not cam:
        return []
    return resolve_kpi_names(cam.get("kpi_ids", []))
