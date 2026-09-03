import logging
from typing import Generator

from sqlmodel import Session, SQLModel, create_engine

logger = logging.getLogger(__name__)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from app.config import settings
        _engine = create_engine(settings.DATABASE_URL)
    return _engine


def get_session() -> Generator[Session, None, None]:
    with Session(get_engine()) as session:
        yield session


def get_session_ctx() -> Session:
    """Non-generator version for use outside FastAPI dependency injection."""
    return Session(get_engine())


def init_db() -> None:
    from .migrations import ensure_tables_exist, run_migrations
    from app.config import settings

    from .models import (
        AuditLog, Camera, Configuration, EmailLog, EmailServer,
        Job, Alert, AlertFrame,
        Organization, Priority, RefreshToken,
        Role, Timezone, User, Zone,
    )

    engine = get_engine()

    ensure_tables_exist(engine)  # always safe: only creates missing tables, never alters/drops
    if settings.MIGRATION_ENABLED:
        run_migrations(engine)
        logger.info("[db] MIGRATION_ENABLED=True - full migration suite applied")
    else:
        logger.info("[db] MIGRATION_ENABLED=False - skipping column/data migrations (new tables still created)")
    _seed_timezones_if_empty(engine)


def _seed_timezones_if_empty(engine) -> None:
    """The `timezones` table is static, global reference data (not
    per-organization) - seeded once, on any startup that finds it empty.
    Never touched by any write endpoint (see app/api/v1/endpoints/timezones.py
    - read-only by design)."""
    from sqlmodel import select

    from .models import Timezone
    from .seed_data.timezones import TIMEZONE_SEED_ROWS

    with Session(engine) as session:
        if session.exec(select(Timezone)).first() is not None:
            return
        for abbreviation, timezone_name, utc_offset, gmt_offset in TIMEZONE_SEED_ROWS:
            session.add(Timezone(
                abbreviation=abbreviation, timezone_name=timezone_name,
                utc_offset=utc_offset, gmt_offset=gmt_offset, enabled=True,
            ))
        session.commit()
        logger.info(f"[db] seeded {len(TIMEZONE_SEED_ROWS)} timezones")
