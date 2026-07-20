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


def _migrate_users(engine) -> None:
    """Phase 0 additions: last_login_at (Roles/Users screens) and mfa_enabled
    (PRD §2.3 — off by default, not required this phase, but cheap to seed now)."""
    _add_columns(engine, "users", [
        ("last_login_at",  "TIMESTAMP", ""),
        ("mfa_enabled",    "BOOLEAN",   " DEFAULT FALSE"),
        ("token_version",  "INTEGER",   " DEFAULT 0"),
    ])


def _migrate_roles(engine) -> None:
    """Some pre-existing databases were created from an earlier, abandoned
    schema attempt (see the leftover `alembic_version` table) that named this
    column `role_name` and never had `is_system`/`org_id` at all; the current
    Role model (app/db/models/role.py) needs all three. Rename + add — safe
    as a no-op once already applied, and only ever run against a table with
    no rows referencing the old shape."""
    inspector = _inspect(engine)
    if "roles" not in inspector.get_table_names():
        return
    columns = {col["name"] for col in inspector.get_columns("roles")}
    if "role_name" in columns and "name" not in columns:
        with engine.begin() as conn:
            conn.execute(_text("ALTER TABLE roles RENAME COLUMN role_name TO name"))
            logger.info("[migrate] roles.role_name renamed to roles.name")

    _add_columns(engine, "roles", [
        ("is_system", "BOOLEAN", " DEFAULT FALSE"),
        ("org_id",    "INTEGER", ""),
        ("zone_ids",  "JSON",    " DEFAULT '[]'"),
    ])


def _migrate_organizations(engine) -> None:
    """Adds default_timezone_id — real FK into the static timezones catalog,
    superseding the old free-text default_timezone column (left in place, unused)."""
    _add_columns(engine, "organizations", [
        ("default_timezone_id", "INTEGER", ""),
    ])


def _migrate_zones(engine) -> None:
    """Adds timezone_id — real FK into the static timezones catalog,
    superseding the old free-text timezone column (left in place, unused)."""
    _add_columns(engine, "zones", [
        ("timezone_id", "INTEGER", ""),
    ])


def _migrate_priorities(engine) -> None:
    """Adds level (severity rank, 1 = highest; 99 = unranked) — needed so
    Phase 2's camera-list enrichment can surface priority_level like the
    frontend expects."""
    _add_columns(engine, "priorities", [
        ("level", "INTEGER", " DEFAULT 99"),
    ])


def _backfill_camera_org_id(engine) -> None:
    """Cameras are seeded from config.json at every startup (see
    app.db.seed_cameras) and predate any org concept, so existing rows have
    org_id=NULL. This platform is single-org-per-deployment (Organization.id
    is always 1), so any camera missing an org_id can only ever belong to
    that one org — backfill it, but only once that org actually exists (a
    fresh deployment with no org registered yet has nothing to backfill to,
    and org_id is a real FK — writing 1 before the row exists would fail)."""
    inspector = _inspect(engine)
    if "cameras" not in inspector.get_table_names() or "organizations" not in inspector.get_table_names():
        return
    with engine.begin() as conn:
        org_exists = conn.execute(_text("SELECT 1 FROM organizations WHERE id = 1")).first()
        if org_exists:
            conn.execute(_text("UPDATE cameras SET org_id = 1 WHERE org_id IS NULL"))


def run_migrations(engine) -> None:
    """Apply all pending schema changes, then run create_all for new tables."""
    _migrate_legacy_stream_columns(engine)
    _migrate_cameras(engine)
    _migrate_users(engine)
    _migrate_roles(engine)
    _migrate_organizations(engine)
    _migrate_zones(engine)
    _migrate_priorities(engine)
    _backfill_camera_org_id(engine)

    SQLModel.metadata.create_all(engine)
    logger.info("[migrate] all migrations applied")
