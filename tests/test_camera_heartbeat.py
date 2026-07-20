"""Phase 4 (camera-offline heartbeat monitor) tests.

_check_once() is a plain sync function — tested directly, no asyncio needed.
stream_recorder_manager.status_for() is monkeypatched so these tests never
touch real RTSP/recording infrastructure.
"""
from sqlmodel import select

from app.db.models import Alert, Camera


def _make_camera(db_session, camera_id, connectivity_status, recording_enabled=True):
    cam = Camera(
        camera_id=camera_id, name=camera_id, recording_enabled=recording_enabled,
        connectivity_status=connectivity_status,
    )
    db_session.add(cam)
    db_session.commit()
    db_session.refresh(cam)
    return cam


def _patch_status(monkeypatch, status: str):
    from app.services import camera_heartbeat
    monkeypatch.setattr(
        camera_heartbeat.stream_recorder_manager, "status_for",
        lambda camera_id: {"status": status, "error": None},
    )


def test_active_to_inactive_creates_offline_alert(db_session, monkeypatch):
    from app.services.camera_heartbeat import _check_once

    _make_camera(db_session, "CAM-HB1", connectivity_status="active")
    _patch_status(monkeypatch, "reconnecting")  # maps to "inactive"

    _check_once()

    cam = db_session.get(Camera, "CAM-HB1")
    db_session.refresh(cam)
    assert cam.connectivity_status == "inactive"

    alerts = db_session.exec(select(Alert).where(Alert.camera_id == "CAM-HB1")).all()
    assert len(alerts) == 1
    assert alerts[0].alert_type == "camera_offline"
    assert alerts[0].job_id is None


def test_pending_to_active_creates_no_alert(db_session, monkeypatch):
    from app.services.camera_heartbeat import _check_once

    _make_camera(db_session, "CAM-HB2", connectivity_status="pending")
    _patch_status(monkeypatch, "connected")  # maps to "active"

    _check_once()

    cam = db_session.get(Camera, "CAM-HB2")
    db_session.refresh(cam)
    assert cam.connectivity_status == "active"

    alerts = db_session.exec(select(Alert).where(Alert.camera_id == "CAM-HB2")).all()
    assert alerts == []


def test_inactive_to_active_recovery_creates_no_alert(db_session, monkeypatch):
    from app.services.camera_heartbeat import _check_once

    _make_camera(db_session, "CAM-HB3", connectivity_status="inactive")
    _patch_status(monkeypatch, "connected")  # maps to "active"

    _check_once()

    cam = db_session.get(Camera, "CAM-HB3")
    db_session.refresh(cam)
    assert cam.connectivity_status == "active"

    alerts = db_session.exec(select(Alert).where(Alert.camera_id == "CAM-HB3")).all()
    assert alerts == []


def test_non_recording_camera_is_never_touched(db_session, monkeypatch):
    from app.services.camera_heartbeat import _check_once

    _make_camera(db_session, "CAM-HB4", connectivity_status="active", recording_enabled=False)
    _patch_status(monkeypatch, "reconnecting")

    _check_once()

    cam = db_session.get(Camera, "CAM-HB4")
    db_session.refresh(cam)
    assert cam.connectivity_status == "active"  # untouched — camera isn't monitored


def test_no_flapping_when_status_unchanged(db_session, monkeypatch):
    from app.services.camera_heartbeat import _check_once

    _make_camera(db_session, "CAM-HB5", connectivity_status="active")
    _patch_status(monkeypatch, "connected")  # maps to "active" — no transition

    _check_once()

    alerts = db_session.exec(select(Alert).where(Alert.camera_id == "CAM-HB5")).all()
    assert alerts == []
