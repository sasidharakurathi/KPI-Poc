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
        ("kpi_model_ids",       "JSON",     " DEFAULT '[]'"),
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


def _migrate_alerts(engine) -> None:
    """Phase 4: adds alerts.camera_id (direct FK) and relaxes job_id/frame_idx
    to nullable — the camera-offline heartbeat monitor creates connectivity
    alerts with no underlying video-processing job and no frame. Backfills
    camera_id from the existing job_id join for every pre-existing row, so
    REST/websocket zone-scoping can filter on Alert.camera_id directly
    without joining Job at query time.

    The nullability relax only runs on Postgres — SQLite (pytest's throwaway
    DB) always gets a fresh, already-nullable schema via create_all, so
    there's nothing to alter there."""
    inspector = _inspect(engine)
    if "alerts" not in inspector.get_table_names():
        return

    _add_columns(engine, "alerts", [
        ("camera_id", "VARCHAR", ""),
    ])

    if engine.dialect.name == "postgresql":
        fresh_columns = {col["name"]: col for col in _inspect(engine).get_columns("alerts")}
        with engine.begin() as conn:
            if fresh_columns.get("job_id", {}).get("nullable") is False:
                conn.execute(_text("ALTER TABLE alerts ALTER COLUMN job_id DROP NOT NULL"))
                logger.info("[migrate] alerts.job_id relaxed to nullable")
            if fresh_columns.get("frame_idx", {}).get("nullable") is False:
                conn.execute(_text("ALTER TABLE alerts ALTER COLUMN frame_idx DROP NOT NULL"))
                logger.info("[migrate] alerts.frame_idx relaxed to nullable")

    if "jobs" in inspector.get_table_names():
        with engine.begin() as conn:
            conn.execute(_text(
                "UPDATE alerts SET camera_id = ("
                "  SELECT jobs.camera_id FROM jobs WHERE jobs.job_id = alerts.job_id"
                ") WHERE alerts.camera_id IS NULL AND alerts.job_id IS NOT NULL"
            ))


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


def _migrate_audit_logs(engine) -> None:
    """Adds summary — a human-readable one-liner (e.g. 'Created camera
    "CAM-01".'), matching the frontend's AuditLogEntry.summary field, which
    the original entity_type/entity_name/before/after shape didn't cover."""
    _add_columns(engine, "audit_logs", [
        ("summary", "VARCHAR", ""),
    ])


def _migrate_kpi_configuration(engine) -> None:
    """Phase 3 (team-built, audited 2026-07-21): adds org_id — the table had
    none at all, so every row was visible regardless of org — and converts
    assigned_models/parameters from manually json.dumps()-serialized TEXT to
    native JSON columns (every other JSON-bearing column in this codebase —
    Alert.extra, Role.permissions, AuditLog.before/after — uses SQLModel's
    JSON type; TEXT meant these fields couldn't be queried, and a malformed
    stored string would silently break response validation instead of
    failing cleanly). The TEXT->JSON conversion only runs on Postgres and is
    safe to run unconditionally: confirmed live that this brand-new table
    had zero real rows at audit time."""
    inspector = _inspect(engine)
    if "kpi_configuration" not in inspector.get_table_names():
        return

    _add_columns(engine, "kpi_configuration", [
        ("org_id", "INTEGER", ""),
        ("description", "VARCHAR", ""),
        ("category", "VARCHAR", ""),
    ])

    if engine.dialect.name == "postgresql":
        fresh_columns = {col["name"]: col for col in _inspect(engine).get_columns("kpi_configuration")}
        with engine.begin() as conn:
            for col_name in ("assigned_models", "parameters"):
                col_type = str(fresh_columns.get(col_name, {}).get("type", "")).upper()
                if "JSON" not in col_type:
                    conn.execute(_text(
                        f"ALTER TABLE kpi_configuration ALTER COLUMN {col_name} "
                        f"TYPE JSON USING {col_name}::json"
                    ))
                    logger.info(f"[migrate] kpi_configuration.{col_name} converted TEXT -> JSON")


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
    _migrate_alerts(engine)
    _migrate_audit_logs(engine)
    _migrate_kpi_configuration(engine)

    SQLModel.metadata.create_all(engine)
    logger.info("[migrate] all migrations applied")
