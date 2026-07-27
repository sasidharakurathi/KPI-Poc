import json
import threading
from pathlib import Path
from typing import Any

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


def update_kpi_config(kpi_class_name: str, updates: dict) -> dict:
    """Merge updates into config.json[kpi_class_name], persist to disk, and reload."""
    global _cache
    with _lock:
        cfg = _load_from_disk()
        block = {**cfg.get(kpi_class_name, {}), **updates}
        cfg[kpi_class_name] = block
        with open(_CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=4)
        _cache = cfg
    return dict(block)


# ── KPI registry (numeric ID → KPI name) ─────────────────────────────────────

def get_kpi_registry() -> dict[str, str]:
    """The kpi_registry mapping: {str(kpi_id): kpi_name}."""
    _ensure_loaded()
    return dict(_cache.get("kpi_registry", {}))


def resolve_kpi_names(kpi_ids: list[int]) -> list[str]:
    """Numeric KPI IDs to implemented KPI names; unmapped IDs are silently skipped."""
    registry = get_kpi_registry()
    seen, names = set(), []
    for kid in kpi_ids:
        name = registry.get(str(kid))
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


# ── Camera seed data ─────────────────────────────────────────────────────────

def get_cameras() -> dict[str, dict]:
    _ensure_loaded()
    return dict(_cache.get("cameras", {}))
