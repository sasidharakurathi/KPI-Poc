"""Idempotent schema migrations.

Run on startup when MIGRATION_ENABLED=True. Every function here is safe to call
repeatedly — it checks current schema state before issuing any DDL.

How to give permission:
  Set MIGRATION_ENABLED=True in .env, then restart the server.
  The flag is False by default so production deployments never migrate silently.
"""
import logging

from sqlmodel import SQLModel
from sqlalchemy import inspect as _inspect, text as _text

logger = logging.getLogger(__name__)


def _add_columns(engine, table_name: str, columns: list[tuple[str, str, str]]) -> None:
    """Add columns to an existing table — no-op for columns that already exist.

    columns: list of (name, sql_type, default_clause) e.g. ("zone_id", "INTEGER", "")
    """
    inspector = _inspect(engine)
    if table_name not in inspector.get_table_names():
        return

    existing = {col["name"] for col in inspector.get_columns(table_name)}
    is_postgres = engine.dialect.name == "postgresql"

    with engine.begin() as conn:
        for name, col_type, default_sql in columns:
            if name in existing:
                continue
            if is_postgres:
                ddl = f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {name} {col_type}{default_sql}"
            else:
                ddl = f"ALTER TABLE {table_name} ADD COLUMN {name} {col_type}{default_sql}"
            conn.execute(_text(ddl))
            logger.info(f"[migrate] {table_name}.{name} added")


def _migrate_cameras(engine) -> None:
    _add_columns(engine, "cameras", [
        ("camera_code",         "VARCHAR",  ""),
        ("mode",                "VARCHAR",  " DEFAULT 'live'"),
        ("connectivity_status", "VARCHAR",  " DEFAULT 'pending'"),
        ("enabled",             "BOOLEAN",  " DEFAULT TRUE"),
        ("org_id",              "INTEGER",  ""),
        ("zone_id",             "INTEGER",  ""),
        ("priority_id",         "INTEGER",  ""),
    ])


def _migrate_legacy_stream_columns(engine) -> None:
    """Columns that may be missing on deployments that predate the RTSP recording feature."""
    _add_columns(engine, "cameras", [
        ("camera_ip",                 "VARCHAR", ""),
        ("rtsp_port",                 "INTEGER", " DEFAULT 554"),
        ("stream_username",           "VARCHAR", ""),
        ("stream_password_encrypted", "VARCHAR", ""),
        ("stream_path",               "VARCHAR", " DEFAULT ''"),
        ("recording_enabled",         "BOOLEAN", " DEFAULT FALSE"),
    ])


def run_migrations(engine) -> None:
    """Apply all pending schema changes, then run create_all for new tables."""
    _migrate_legacy_stream_columns(engine)
    _migrate_cameras(engine)

    SQLModel.metadata.create_all(engine)
    logger.info("[migrate] all migrations applied")
