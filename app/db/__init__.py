"""Public interface for the db package.

Existing code (pipeline.py, stream_recorder.py, notifications.py, …) imports
from `app.db` directly.  All those imports continue to work unchanged.
"""
import json
import logging
from datetime import datetime
from typing import Any, Optional

from sqlmodel import Session, select
from sqlalchemy import func as _func

from .engine import get_engine, get_session, get_session_ctx, init_db
from .models import (
    Alert, AlertFrame, AuditLog, Camera, Configuration,
    EmailLog, EmailServer, Job,
    KpiModelCatalog, Organization, Priority, RefreshToken,
    Role, Timezone, User, Zone,
)

logger = logging.getLogger(__name__)

__all__ = [
    # engine helpers
    "get_engine", "get_session", "get_session_ctx", "init_db",
    # models
    "Alert", "AlertFrame", "AuditLog", "Camera", "Configuration",
    "EmailLog", "EmailServer", "Job",
    "KpiModelCatalog", "Organization", "Priority", "RefreshToken",
    "Role", "Timezone", "User", "Zone",
    # write helpers
    "create_alert", "add_alert_frames", "upsert_job", "update_job", "get_job",
    # camera helpers
    "seed_cameras", "list_cameras", "get_camera", "create_camera",
    "update_camera", "delete_camera",
    # config helpers
    "get_config", "get_config_updated_at", "set_config", "list_configs",
    # email log helpers
    "create_email_log", "query_email_logs",
    # alert query
    "query_alerts", "get_alert",
]


# ── Write helpers ─────────────────────────────────────────────────────────────

def create_alert(
    job_id: str,
    kpi_name: str,
    alert_type: str,
    frame_idx: int,
    confidence: float = 1.0,
    extra: Optional[dict] = None,
) -> int:
    with get_session_ctx() as session:
        alert = Alert(
            job_id=job_id, kpi_name=kpi_name, alert_type=alert_type,
            frame_idx=frame_idx, confidence=confidence, extra=extra,
            created_at=datetime.utcnow(),
        )
        session.add(alert)
        session.commit()
        session.refresh(alert)
        return alert.id  # type: ignore[return-value]


def add_alert_frames(
    alert_id: int,
    records: list[tuple[int, int, str, Optional[str]]],
) -> None:
    with get_session_ctx() as session:
        for pos, fidx, path, labeled_path in records:
            session.add(AlertFrame(
                alert_id=alert_id, position=pos,
                frame_idx=fidx, path=path, labeled_path=labeled_path,
            ))
        session.commit()


def upsert_job(
    job_id: str,
    filename: str,
    video_path: str,
    camera_id: Optional[str] = None,
    camera_name: Optional[str] = None,
    kpis_running: Optional[list] = None,
) -> Job:
    with get_session_ctx() as session:
        job = Job(
            job_id=job_id, filename=filename, video_path=video_path,
            camera_id=camera_id, camera_name=camera_name,
            status="pending", kpis_running=kpis_running or [],
            created_at=datetime.utcnow(),
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        return job


def update_job(
    job_id: str,
    status: str,
    kpi_results: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    with get_session_ctx() as session:
        job = session.get(Job, job_id)
        if not job:
            return
        job.status = status
        if status in ("completed", "failed"):
            job.completed_at = datetime.utcnow()
        if kpi_results is not None:
            job.kpi_results = kpi_results
        if error is not None:
            job.error = error
        session.add(job)
        session.commit()


def get_job(job_id: str) -> Optional[Job]:
    with get_session_ctx() as session:
        return session.get(Job, job_id)


# ── Camera helpers ────────────────────────────────────────────────────────────

def seed_cameras(cameras: dict[str, dict]) -> None:
    with get_session_ctx() as session:
        existing = set(session.exec(select(Camera.camera_id)).all())
        # Single-org-per-deployment: stamp org_id so newly-seeded cameras are
        # visible through the org-scoped /api/cameras endpoints right away,
        # instead of silently landing with org_id=NULL (see also
        # migrations._backfill_camera_org_id for rows from before this org existed).
        org = session.get(Organization, 1)
        for camera_id, cam in cameras.items():
            if camera_id in existing:
                continue
            session.add(Camera(
                camera_id=camera_id,
                name=cam["name"],
                zone=cam.get("zone", ""),
                priority=cam.get("priority", "medium"),
                kpi_ids=cam.get("kpi_ids", []),
                org_id=org.id if org else None,
            ))
        session.commit()


def list_cameras() -> list[Camera]:
    with get_session_ctx() as session:
        return list(session.exec(select(Camera).order_by(Camera.camera_id)).all())  # type: ignore[arg-type]


def get_camera(camera_id: str) -> Optional[Camera]:
    with get_session_ctx() as session:
        return session.get(Camera, camera_id)


def create_camera(
    camera_id: str,
    name: str,
    zone: str = "",
    priority: str = "medium",
    kpi_ids: Optional[list[int]] = None,
    camera_ip: Optional[str] = None,
    rtsp_port: int = 554,
    stream_username: Optional[str] = None,
    stream_password_encrypted: Optional[str] = None,
    stream_path: str = "",
    recording_enabled: bool = False,
) -> Camera:
    with get_session_ctx() as session:
        if session.get(Camera, camera_id):
            raise ValueError(f"Camera '{camera_id}' already exists.")
        camera = Camera(
            camera_id=camera_id, name=name, zone=zone,
            priority=priority, kpi_ids=kpi_ids or [],
            camera_ip=camera_ip, rtsp_port=rtsp_port,
            stream_username=stream_username,
            stream_password_encrypted=stream_password_encrypted,
            stream_path=stream_path, recording_enabled=recording_enabled,
        )
        session.add(camera)
        session.commit()
        session.refresh(camera)
        return camera


def update_camera(
    camera_id: str,
    name: Optional[str] = None,
    zone: Optional[str] = None,
    priority: Optional[str] = None,
    kpi_ids: Optional[list[int]] = None,
    camera_ip: Optional[str] = None,
    rtsp_port: Optional[int] = None,
    stream_username: Optional[str] = None,
    stream_password_encrypted: Optional[str] = None,
    stream_path: Optional[str] = None,
    recording_enabled: Optional[bool] = None,
) -> Optional[Camera]:
    with get_session_ctx() as session:
        camera = session.get(Camera, camera_id)
        if not camera:
            return None
        if name is not None:
            camera.name = name
        if zone is not None:
            camera.zone = zone
        if priority is not None:
            camera.priority = priority
        if kpi_ids is not None:
            camera.kpi_ids = kpi_ids
        if camera_ip is not None:
            camera.camera_ip = camera_ip
        if rtsp_port is not None:
            camera.rtsp_port = rtsp_port
        if stream_username is not None:
            camera.stream_username = stream_username
        if stream_password_encrypted is not None:
            camera.stream_password_encrypted = stream_password_encrypted
        if stream_path is not None:
            camera.stream_path = stream_path
        if recording_enabled is not None:
            camera.recording_enabled = recording_enabled
        camera.updated_at = datetime.utcnow()
        session.add(camera)
        session.commit()
        session.refresh(camera)
        return camera


def delete_camera(camera_id: str) -> bool:
    with get_session_ctx() as session:
        camera = session.get(Camera, camera_id)
        if not camera:
            return False
        session.delete(camera)
        session.commit()
        return True


# ── Generic configuration store ───────────────────────────────────────────────

def get_config(name: str) -> Optional[dict]:
    with get_session_ctx() as session:
        row = session.get(Configuration, name)
        if not row or not row.value:
            return None
        try:
            return json.loads(row.value)
        except (json.JSONDecodeError, TypeError):
            logger.warning("[config] '%s' has invalid JSON — ignoring", name)
            return None


def get_config_updated_at(name: str) -> Optional[datetime]:
    with get_session_ctx() as session:
        row = session.get(Configuration, name)
        return row.updated_at if row else None


def set_config(name: str, value: dict) -> dict:
    with get_session_ctx() as session:
        row = session.get(Configuration, name)
        if not row:
            row = Configuration(name=name)
        row.value = json.dumps(value)
        row.updated_at = datetime.utcnow()
        session.add(row)
        session.commit()
        return value


def list_configs() -> list[Configuration]:
    with get_session_ctx() as session:
        return list(session.exec(select(Configuration).order_by(Configuration.name)).all())  # type: ignore[arg-type]


# ── Email log helpers ─────────────────────────────────────────────────────────

def create_email_log(
    status: str,
    subject: str = "",
    recipients: Optional[list[str]] = None,
    alert_id: Optional[int] = None,
    kpi_name: Optional[str] = None,
    alert_type: Optional[str] = None,
    camera_id: Optional[str] = None,
    camera_name: Optional[str] = None,
    error: Optional[str] = None,
) -> int:
    with get_session_ctx() as session:
        log = EmailLog(
            status=status, subject=subject, recipients=recipients or [],
            alert_id=alert_id, kpi_name=kpi_name, alert_type=alert_type,
            camera_id=camera_id, camera_name=camera_name, error=error,
        )
        session.add(log)
        session.commit()
        session.refresh(log)
        return log.id  # type: ignore[return-value]


_EMAIL_LOG_SORT_COLUMNS: dict[str, Any] = {
    "created_at": EmailLog.created_at,
    "kpi_name":   EmailLog.kpi_name,
    "status":     EmailLog.status,
    "camera_id":  EmailLog.camera_id,
}


def query_email_logs(
    status: Optional[str] = None,
    kpi_name: Optional[str] = None,
    camera_id: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    sort_by: str = "created_at",
    sort_dir: str = "desc",
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[EmailLog], int]:
    with get_session_ctx() as session:
        stmt = select(EmailLog)
        count_stmt = select(_func.count()).select_from(EmailLog)

        if status:
            stmt = stmt.where(EmailLog.status == status)
            count_stmt = count_stmt.where(EmailLog.status == status)
        if kpi_name:
            stmt = stmt.where(EmailLog.kpi_name == kpi_name)
            count_stmt = count_stmt.where(EmailLog.kpi_name == kpi_name)
        if camera_id:
            stmt = stmt.where(EmailLog.camera_id.ilike(f"%{camera_id}%"))
            count_stmt = count_stmt.where(EmailLog.camera_id.ilike(f"%{camera_id}%"))
        if date_from:
            stmt = stmt.where(EmailLog.created_at >= date_from)
            count_stmt = count_stmt.where(EmailLog.created_at >= date_from)
        if date_to:
            stmt = stmt.where(EmailLog.created_at <= date_to)
            count_stmt = count_stmt.where(EmailLog.created_at <= date_to)

        col = _EMAIL_LOG_SORT_COLUMNS.get(sort_by, EmailLog.created_at)
        stmt = stmt.order_by(col.desc() if sort_dir == "desc" else col.asc())
        stmt = stmt.offset(offset).limit(limit)

        total = session.exec(count_stmt).one()
        rows = list(session.exec(stmt).all())
        return rows, total


# ── Alert query helpers ───────────────────────────────────────────────────────

def _alert_to_dict(alert: Alert, session: Session) -> dict:
    frames = session.exec(
        select(AlertFrame)
        .where(AlertFrame.alert_id == alert.id)
        .order_by(AlertFrame.position)  # type: ignore[arg-type]
    ).all()
    job = session.get(Job, alert.job_id)
    return {
        "id": alert.id,
        "job_id": alert.job_id,
        "camera_id": job.camera_id if job else None,
        "camera_name": job.camera_name if job else None,
        "kpi_name": alert.kpi_name,
        "alert_type": alert.alert_type,
        "frame_idx": alert.frame_idx,
        "confidence": alert.confidence,
        "extra": alert.extra,
        "created_at": alert.created_at,
        "frames": [
            {
                "id": f.id,
                "position": f.position,
                "frame_idx": f.frame_idx,
                "path": f.path,
                "labeled_path": f.labeled_path,
            }
            for f in frames
        ],
    }


_ALERT_SORT_COLUMNS: dict[str, Any] = {
    "created_at": Alert.created_at,
    "kpi_name":   Alert.kpi_name,
    "alert_type": Alert.alert_type,
    "confidence": Alert.confidence,
}


def query_alerts(
    job_id: Optional[str] = None,
    kpi_name: Optional[str] = None,
    camera_id: Optional[str] = None,
    alert_type: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    sort_by: str = "created_at",
    sort_dir: str = "desc",
    limit: int = 200,
    offset: int = 0,
) -> tuple[list[dict], int]:
    with get_session_ctx() as session:
        stmt = select(Alert)
        count_stmt = select(_func.count()).select_from(Alert)

        if camera_id:
            stmt = stmt.join(Job, Job.job_id == Alert.job_id)  # type: ignore[arg-type]
            count_stmt = count_stmt.join(Job, Job.job_id == Alert.job_id)  # type: ignore[arg-type]
            stmt = stmt.where(Job.camera_id == camera_id)
            count_stmt = count_stmt.where(Job.camera_id == camera_id)
        if job_id:
            stmt = stmt.where(Alert.job_id == job_id)
            count_stmt = count_stmt.where(Alert.job_id == job_id)
        if kpi_name:
            stmt = stmt.where(Alert.kpi_name == kpi_name)
            count_stmt = count_stmt.where(Alert.kpi_name == kpi_name)
        if alert_type:
            stmt = stmt.where(Alert.alert_type.ilike(f"%{alert_type}%"))
            count_stmt = count_stmt.where(Alert.alert_type.ilike(f"%{alert_type}%"))
        if date_from:
            stmt = stmt.where(Alert.created_at >= date_from)
            count_stmt = count_stmt.where(Alert.created_at >= date_from)
        if date_to:
            stmt = stmt.where(Alert.created_at <= date_to)
            count_stmt = count_stmt.where(Alert.created_at <= date_to)

        col = _ALERT_SORT_COLUMNS.get(sort_by, Alert.created_at)
        stmt = stmt.order_by(col.desc() if sort_dir == "desc" else col.asc())
        stmt = stmt.offset(offset).limit(limit)

        total = session.exec(count_stmt).one()
        rows = [_alert_to_dict(a, session) for a in session.exec(stmt).all()]
        return rows, total


def get_alert(alert_id: int) -> Optional[dict]:
    with get_session_ctx() as session:
        alert = session.get(Alert, alert_id)
        if not alert:
            return None
        return _alert_to_dict(alert, session)
