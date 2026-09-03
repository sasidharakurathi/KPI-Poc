"""Idempotent schema migrations.

Two different things happen at startup (see app.db.engine.init_db), gated
differently:
  ensure_tables_exist() - creates any table missing from the DB, from the
    current SQLModel definitions. Always runs, unconditionally: a brand-new
    table already has every column its model declares (e.g. a fresh
    `cameras` table already has latitude/longitude), so this alone is
    enough to bring an empty database fully up to date. Purely additive -
    never alters or drops an existing table - so it's always safe.
  run_migrations() - ALTERs existing tables (adding columns a table
    created before that column existed is missing) and backfills/dedupes
    data on them. Only runs when MIGRATION_ENABLED=True, since unlike
    table creation this can rewrite real data on a live deployment (e.g.
    _backfill_camera_org_id, _migrate_kpi_catalog_to_global's dedup
    delete) and every function here is safe to call repeatedly, but not
    something that should happen silently on every restart.

How to give permission:
  Set MIGRATION_ENABLED=True in .env, then restart the server.
  The flag is False by default so production deployments never migrate silently.
"""
import logging

from sqlmodel import SQLModel
from sqlalchemy import inspect as _inspect, text as _text

logger = logging.getLogger(__name__)


def _add_columns(engine, table_name: str, columns: list[tuple[str, str, str]]) -> None:
    """Add columns to an existing table - no-op for columns that already exist.

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
        ("latitude",            "FLOAT",    ""),
        ("longitude",           "FLOAT",    ""),
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
    (PRD §2.3 - off by default, not required this phase, but cheap to seed now)."""
    _add_columns(engine, "users", [
        ("last_login_at",  "TIMESTAMP", ""),
        ("mfa_enabled",    "BOOLEAN",   " DEFAULT FALSE"),
        ("token_version",  "INTEGER",   " DEFAULT 0"),
    ])


def _migrate_roles(engine) -> None:
    """Some pre-existing databases were created from an earlier, abandoned
    schema attempt (see the leftover `alembic_version` table) that named this
    column `role_name` and never had `is_system`/`org_id` at all; the current
    Role model (app/db/models/role.py) needs all three. Rename + add - safe
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
    """Adds default_timezone_id - real FK into the static timezones catalog,
    superseding the old free-text default_timezone column (left in place, unused)."""
    _add_columns(engine, "organizations", [
        ("default_timezone_id", "INTEGER", ""),
    ])


def _migrate_zones(engine) -> None:
    """Adds timezone_id - real FK into the static timezones catalog,
    superseding the old free-text timezone column (left in place, unused)."""
    _add_columns(engine, "zones", [
        ("timezone_id", "INTEGER", ""),
    ])


def _migrate_priorities(engine) -> None:
    """Adds level (severity rank, 1 = highest; 99 = unranked) - needed so
    Phase 2's camera-list enrichment can surface priority_level like the
    frontend expects."""
    _add_columns(engine, "priorities", [
        ("level", "INTEGER", " DEFAULT 99"),
    ])


def _migrate_alerts(engine) -> None:
    """Phase 4: adds alerts.camera_id (direct FK) and relaxes job_id/frame_idx
    to nullable - the camera-offline heartbeat monitor creates connectivity
    alerts with no underlying video-processing job and no frame. Backfills
    camera_id from the existing job_id join for every pre-existing row, so
    REST/websocket zone-scoping can filter on Alert.camera_id directly
    without joining Job at query time.

    The nullability relax only runs on Postgres - SQLite (pytest's throwaway
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


def _migrate_alert_org_id(engine) -> None:
    """Adds alerts.org_id and backfills every existing row from its camera
    (Alert.camera_id -> Camera.org_id) - the direct link that makes org-
    scoped alert filtering possible without a join at query time. A handful
    of legacy rows might have camera_id but no matching camera anymore (a
    since-deleted camera); those are left with org_id=NULL and are simply
    invisible to org-scoped queries going forward, same as camera-less
    connectivity alerts already are to zone-scoped ones."""
    inspector = _inspect(engine)
    if "alerts" not in inspector.get_table_names():
        return

    _add_columns(engine, "alerts", [
        ("org_id", "INTEGER", ""),
    ])

    if "cameras" in inspector.get_table_names():
        with engine.begin() as conn:
            conn.execute(_text(
                "UPDATE alerts SET org_id = ("
                "  SELECT cameras.org_id FROM cameras WHERE cameras.camera_id = alerts.camera_id"
                ") WHERE alerts.org_id IS NULL AND alerts.camera_id IS NOT NULL"
            ))


def _backfill_camera_org_id(engine) -> None:
    """Cameras are seeded from config.json at every startup (see
    app.db.seed_cameras) and predate any org concept, so existing rows have
    org_id=NULL. This is a one-time backfill for those legacy rows: they're
    attributed to the first organization ever registered on this deployment
    (lowest id), the same convention app.db.seed_cameras uses going forward
    for newly-seeded cameras - config.json still represents one physical
    camera fleet, regardless of how many organizations now exist on top of
    it. A no-op if no organization has been registered yet."""
    inspector = _inspect(engine)
    if "cameras" not in inspector.get_table_names() or "organizations" not in inspector.get_table_names():
        return
    with engine.begin() as conn:
        first_org = conn.execute(_text("SELECT id FROM organizations ORDER BY id LIMIT 1")).first()
        if first_org:
            conn.execute(_text("UPDATE cameras SET org_id = :org_id WHERE org_id IS NULL"), {"org_id": first_org[0]})


def _migrate_audit_logs(engine) -> None:
    """Adds summary - a human-readable one-liner (e.g. 'Created camera
    "CAM-01".'), matching the frontend's AuditLogEntry.summary field, which
    the original entity_type/entity_name/before/after shape didn't cover."""
    _add_columns(engine, "audit_logs", [
        ("summary", "VARCHAR", ""),
    ])


def _rescope_unique_constraint_to_org(engine, table_name: str, column: str, org_column: str = "org_id") -> None:
    """Replaces a single-column UNIQUE(column) constraint with a composite
    UNIQUE(org_column, column) one.

    Several org-scoped tables (Priority.name, EmailServer.label,
    Role.name) were originally declared with a
    column-level `unique=True` in SQLModel - a global constraint - even
    though every endpoint's own duplicate-check has always been scoped to
    org_id. That mismatch was invisible on a single-org deployment (there
    was only ever one org to collide within) and becomes a real bug the
    moment a second organization exists: its first "Critical" priority (or
    "Owner" role, seeded automatically at registration) would collide with
    the first org's row and fail with a misleading 409/500.

    Postgres auto-names a column-level UNIQUE constraint `<table>_<column>_key`,
    but we discover it via the inspector instead of hardcoding that, in case
    a given deployment's constraint was named differently.
    """
    inspector = _inspect(engine)
    if table_name not in inspector.get_table_names():
        return

    existing = inspector.get_unique_constraints(table_name)
    old_constraint = next(
        (uc for uc in existing if uc["column_names"] == [column] and uc.get("name")), None,
    )
    composite_present = any(
        set(uc["column_names"]) == {org_column, column} for uc in existing
    )

    with engine.begin() as conn:
        if old_constraint is not None:
            conn.execute(_text(f'ALTER TABLE {table_name} DROP CONSTRAINT "{old_constraint["name"]}"'))
            logger.info(f"[migrate] {table_name}: dropped global unique constraint on {column}")
        if not composite_present:
            constraint_name = f"uq_{table_name}_{org_column}_{column}"
            conn.execute(_text(
                f'ALTER TABLE {table_name} ADD CONSTRAINT "{constraint_name}" UNIQUE ({org_column}, {column})'
            ))
            logger.info(f"[migrate] {table_name}: added composite unique constraint on ({org_column}, {column})")


def _migrate_multi_org_unique_constraints(engine) -> None:
    """SQLite (pytest's throwaway DB) always gets the composite constraint
    fresh via create_all - this only needs to run against Postgres, where
    these tables may already exist with the old global constraint."""
    if engine.dialect.name != "postgresql":
        return
    _rescope_unique_constraint_to_org(engine, "priorities", "name")
    _rescope_unique_constraint_to_org(engine, "email_servers", "label")
    _rescope_unique_constraint_to_org(engine, "roles", "name")


def _dedupe_and_make_globally_unique(engine, table_name: str, column: str) -> None:
    """The reverse of _rescope_unique_constraint_to_org: drops a composite
    (org_id, column) unique constraint (or adds none if there wasn't one)
    and replaces it with a plain UNIQUE(column) - used for tables that used
    to be per-org but are now shared across every organization
    (kpi_configuration).

    If two different orgs already created a row with the same value (e.g.
    two "Fire & Smoke Model" rows), a straight ADD CONSTRAINT would fail -
    so any duplicates are deduped first, keeping the lowest id (oldest row)
    and deleting the rest. This is a real, logged data change, not just a
    schema change; duplicates are expected to be rare-to-nonexistent since
    this only matters for deployments that had multiple organizations
    creating same-named catalog entries independently before this migration.
    """
    inspector = _inspect(engine)
    if table_name not in inspector.get_table_names():
        return

    with engine.begin() as conn:
        dupes = conn.execute(_text(
            f"SELECT {column}, array_agg(id ORDER BY id) FROM {table_name} "
            f"GROUP BY {column} HAVING COUNT(*) > 1"
        )).all()
        for value, ids in dupes:
            keep, drop = ids[0], ids[1:]
            conn.execute(_text(f"DELETE FROM {table_name} WHERE id = ANY(:drop)"), {"drop": drop})
            logger.warning(
                f"[migrate] {table_name}: deduped {len(drop)} row(s) with {column}={value!r} "
                f"(kept id={keep}, dropped {drop}) while making {column} globally unique"
            )

        existing = _inspect(engine).get_unique_constraints(table_name)
        old_composite = next(
            (uc for uc in existing if set(uc["column_names"]) == {"org_id", column} and uc.get("name")), None,
        )
        already_global = any(uc["column_names"] == [column] for uc in existing)

        if old_composite is not None:
            conn.execute(_text(f'ALTER TABLE {table_name} DROP CONSTRAINT "{old_composite["name"]}"'))
            logger.info(f"[migrate] {table_name}: dropped composite (org_id, {column}) unique constraint")
        if not already_global:
            constraint_name = f"uq_{table_name}_{column}"
            conn.execute(_text(
                f'ALTER TABLE {table_name} ADD CONSTRAINT "{constraint_name}" UNIQUE ({column})'
            ))
            logger.info(f"[migrate] {table_name}: added global unique constraint on {column}")


def _migrate_kpi_catalog_to_global(engine) -> None:
    """kpi_configuration became shared across every organization instead of
    per-org - see its model's docstring. Only needs to run on Postgres;
    SQLite (pytest) always gets the global constraint fresh via create_all."""
    if engine.dialect.name != "postgresql":
        return
    _dedupe_and_make_globally_unique(engine, "kpi_configuration", "kpi_name")


def _migrate_kpi_configuration(engine) -> None:
    """Phase 3 (team-built, audited 2026-07-21): adds org_id - the table had
    none at all, so every row was visible regardless of org - and converts
    assigned_models/parameters from manually json.dumps()-serialized TEXT to
    native JSON columns (every other JSON-bearing column in this codebase -
    Alert.extra, Role.permissions, AuditLog.before/after - uses SQLModel's
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


def ensure_tables_exist(engine) -> None:
    """Creates any table defined in the SQLModel metadata that doesn't
    exist yet - safe to call unconditionally, regardless of
    MIGRATION_ENABLED (never alters or drops an existing table)."""
    SQLModel.metadata.create_all(engine)


def run_migrations(engine) -> None:
    """ALTERs existing tables and backfills/dedupes data on them. Only
    called when MIGRATION_ENABLED=True - see app.db.engine.init_db."""
    _migrate_legacy_stream_columns(engine)
    _migrate_cameras(engine)
    _migrate_users(engine)
    _migrate_roles(engine)
    _migrate_organizations(engine)
    _migrate_zones(engine)
    _migrate_priorities(engine)
    _backfill_camera_org_id(engine)
    _migrate_alerts(engine)
    _migrate_alert_org_id(engine)
    _migrate_audit_logs(engine)
    _migrate_kpi_configuration(engine)
    _migrate_multi_org_unique_constraints(engine)
    _migrate_kpi_catalog_to_global(engine)

    logger.info("[migrate] all migrations applied")
