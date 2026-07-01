"""
SQLModel-based persistence layer.

Tables:
  jobs         — one row per uploaded video job
  alerts       — one detection event per row
  alert_frames — the 8 (before+anchor+after) raw frame paths per alert,
                 plus optional labeled_path for the anchor frame (dev mode)
"""
import logging
from datetime import datetime
from typing import Optional

from sqlmodel import Column, Field, Relationship, Session, SQLModel, create_engine, select
from sqlalchemy import JSON as _JSON

logger = logging.getLogger(__name__)

_engine = None


# ── ORM models ────────────────────────────────────────────────────────────────

class Job(SQLModel, table=True):
    __tablename__ = "jobs"  # type: ignore[assignment]

    job_id: str = Field(primary_key=True)
    filename: str
    video_path: str
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    status: str = Field(default="pending", index=True)
    kpis_running: list = Field(default_factory=list, sa_column=Column(_JSON))
    kpi_results: Optional[dict] = Field(default=None, sa_column=Column(_JSON))
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    alerts: list["Alert"] = Relationship(back_populates="job")


class Alert(SQLModel, table=True):
    __tablename__ = "alerts"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="jobs.job_id", index=True)
    kpi_name: str
    alert_type: str
    frame_idx: int
    confidence: float = 1.0
    extra: Optional[dict] = Field(default=None, sa_column=Column(_JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    job: Optional[Job] = Relationship(back_populates="alerts")
    frames: list["AlertFrame"] = Relationship(back_populates="alert")


class AlertFrame(SQLModel, table=True):
    __tablename__ = "alert_frames"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    alert_id: int = Field(foreign_key="alerts.id", index=True)
    position: int
    frame_idx: int
    path: str
    labeled_path: Optional[str] = None

    alert: Optional[Alert] = Relationship(back_populates="frames")


class Camera(SQLModel, table=True):
    __tablename__ = "cameras"  # type: ignore[assignment]

    camera_id: str = Field(primary_key=True)
    name: str
    zone: str = ""
    priority: str = "medium"
    kpi_ids: list = Field(default_factory=list, sa_column=Column(_JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Engine ────────────────────────────────────────────────────────────────────

def get_engine():
    global _engine
    if _engine is None:
        from .config import settings
        _engine = create_engine(settings.DATABASE_URL)
    return _engine


def init_db() -> None:
    SQLModel.metadata.create_all(get_engine())


def get_session() -> Session:
    return Session(get_engine())


# ── Write helpers ─────────────────────────────────────────────────────────────

def create_alert(
    job_id: str,
    kpi_name: str,
    alert_type: str,
    frame_idx: int,
    confidence: float = 1.0,
    extra: Optional[dict] = None,
) -> int:
    with get_session() as session:
        alert = Alert(
            job_id=job_id,
            kpi_name=kpi_name,
            alert_type=alert_type,
            frame_idx=frame_idx,
            confidence=confidence,
            extra=extra,
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
    """records: list of (position, frame_idx, path, labeled_path)"""
    with get_session() as session:
        for pos, fidx, path, labeled_path in records:
            session.add(AlertFrame(
                alert_id=alert_id,
                position=pos,
                frame_idx=fidx,
                path=path,
                labeled_path=labeled_path,
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
    with get_session() as session:
        job = Job(
            job_id=job_id,
            filename=filename,
            video_path=video_path,
            camera_id=camera_id,
            camera_name=camera_name,
            status="pending",
            kpis_running=kpis_running or [],
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
    with get_session() as session:
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
    with get_session() as session:
        return session.get(Job, job_id)


# ── Camera helpers ───────────────────────────────────────────────────────────

def seed_cameras(cameras: dict[str, dict]) -> None:
    """Insert cameras from a {camera_id: {...}} dict, skipping ones that already exist."""
    with get_session() as session:
        existing = set(session.exec(select(Camera.camera_id)).all())
        for camera_id, cam in cameras.items():
            if camera_id in existing:
                continue
            session.add(Camera(
                camera_id=camera_id,
                name=cam["name"],
                zone=cam.get("zone", ""),
                priority=cam.get("priority", "medium"),
                kpi_ids=cam.get("kpi_ids", []),
            ))
        session.commit()


def list_cameras() -> list[Camera]:
    with get_session() as session:
        return list(session.exec(select(Camera).order_by(Camera.camera_id)).all())  # type: ignore[arg-type]


def get_camera(camera_id: str) -> Optional[Camera]:
    with get_session() as session:
        return session.get(Camera, camera_id)


def create_camera(
    camera_id: str,
    name: str,
    zone: str = "",
    priority: str = "medium",
    kpi_ids: Optional[list[int]] = None,
) -> Camera:
    with get_session() as session:
        if session.get(Camera, camera_id):
            raise ValueError(f"Camera '{camera_id}' already exists.")
        camera = Camera(
            camera_id=camera_id, name=name, zone=zone,
            priority=priority, kpi_ids=kpi_ids or [],
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
) -> Optional[Camera]:
    with get_session() as session:
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
        camera.updated_at = datetime.utcnow()
        session.add(camera)
        session.commit()
        session.refresh(camera)
        return camera


def delete_camera(camera_id: str) -> bool:
    with get_session() as session:
        camera = session.get(Camera, camera_id)
        if not camera:
            return False
        session.delete(camera)
        session.commit()
        return True


# ── Read helpers ──────────────────────────────────────────────────────────────

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


def query_alerts(
    job_id: Optional[str] = None,
    kpi_name: Optional[str] = None,
    camera_id: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    with get_session() as session:
        stmt = select(Alert)
        if job_id:
            stmt = stmt.where(Alert.job_id == job_id)
        if kpi_name:
            stmt = stmt.where(Alert.kpi_name == kpi_name)
        if camera_id:
            stmt = stmt.join(Job, Job.job_id == Alert.job_id).where(Job.camera_id == camera_id)  # type: ignore[arg-type]
        stmt = stmt.order_by(Alert.created_at.desc()).limit(limit)  # type: ignore[union-attr]
        return [_alert_to_dict(a, session) for a in session.exec(stmt).all()]


def get_alert(alert_id: int) -> Optional[dict]:
    with get_session() as session:
        alert = session.get(Alert, alert_id)
        if not alert:
            return None
        return _alert_to_dict(alert, session)
