"""Camera business logic — thin layer between the API router and the db helpers.

Phase 0 devs: add domain rules here (e.g. "can't delete a camera with active jobs")
rather than in the route handler.
"""
from typing import Optional

from app import db
from app.email_crypto import EmailCryptoNotConfigured, encrypt_secret
from app.stream_recorder import stream_recorder_manager


def build_camera_info(cam: db.Camera) -> dict:
    from app.config_loader import get_kpi_registry, list_registered_names
    from app.main import _KPI_LABELS

    registry = get_kpi_registry()
    implemented_names = set(list_registered_names())
    kpis_detail = [
        {
            "kpi_id": kid,
            "kpi_label": _KPI_LABELS.get(kid, f"KPI {kid}"),
            "implemented": registry.get(str(kid)) in implemented_names,
        }
        for kid in cam.kpi_ids
    ]
    stream_status = stream_recorder_manager.status_for(cam.camera_id)
    return {
        "camera_id": cam.camera_id,
        "name": cam.name,
        "zone": cam.zone,
        "priority": cam.priority,
        "kpi_ids": cam.kpi_ids,
        "kpis": kpis_detail,
        "camera_ip": cam.camera_ip,
        "rtsp_port": cam.rtsp_port,
        "stream_username": cam.stream_username,
        "stream_path": cam.stream_path,
        "stream_password_set": bool(cam.stream_password_encrypted),
        "recording_enabled": cam.recording_enabled,
        "stream_status": stream_status["status"],
        "stream_error": stream_status["error"],
    }


def encrypt_camera_password(plain: Optional[str]) -> Optional[str]:
    if not plain:
        return None
    try:
        return encrypt_secret(plain)
    except EmailCryptoNotConfigured as e:
        raise ValueError(str(e))
