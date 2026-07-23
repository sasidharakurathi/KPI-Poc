"""Populates every reference/config table an organization needs for realistic
demo data, with proper foreign-key references between them — the pieces
scripts/seed_kpi_catalog.py and scripts/seed_alerts_from_storage.py don't
cover. Run after both of those (kpi_configuration and alerts/jobs should
already exist) so this script can link against them.

Tables touched, and why:
  priorities          new rows (Critical/High/Medium/Low) — 0 existed before
  zones               new rows, grouped from real camera names (rig/vessel
                       areas: Wheel House, Engine Room, Crane Area, ...)
  cameras             UPDATED in place: zone_id/priority_id/kpi_model_ids —
                       every existing camera had all three NULL/empty
  kpi_model_catalog   new rows, one per registered KPI, using each
                       detector's REAL model_path/confidence from config.json
                       (app.config_loader) — not made up
  kpi_configuration   UPDATED in place: assigned_models now references the
                       kpi_model_catalog rows above instead of staying empty
  roles               2 new sample roles (Operator, Zone Guard) alongside
                       the existing Owner/Viewer
  users               new users for those roles
  email_logs          new rows referencing existing alerts (a plausible
                       subset — not every alert necessarily emails)

Tables deliberately NOT touched, and why:
  audit_logs          a factual record of real actions taken through the API
                       (app.services.audit_service) — fabricating entries
                       here would make it lie about what actually happened.
  refresh_tokens       live session state, not reference/demo data.
  configurations       internal key-value system state (e.g. the default
                       timezone), not something to seed dummy rows into.
  timezones            already a complete, correct static catalog (200 rows).
  email_servers        already has a working bootstrap default; adding fake
                       SMTP servers risks someone mistaking one for usable.
  organizations        already has real registered org(s) — this script
                       only ever operates within an existing one.

Idempotent by default: cameras/kpi_configuration are only updated where
currently empty (pass --force to overwrite), and every insert checks for an
existing row with the same name/username first. Safe to re-run.

Usage:
    python scripts/seed_reference_data.py                # first org, real run
    python scripts/seed_reference_data.py --org-id 2
    python scripts/seed_reference_data.py --dry-run
    python scripts/seed_reference_data.py --force         # overwrite cameras'/
                                                            # kpi_configuration's
                                                            # existing zone/priority/
                                                            # kpi_model_ids/assigned_models
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlmodel import Session, select

from app.config_loader import get_kpi_config
from app.core.security import hash_password
from app.db.engine import get_engine
from app.db.models import (
    Alert, Camera, EmailLog, KpiModelCatalog, Organization, Priority, Role, User, Zone,
)
from app.db.models.kpi_configuration import KPIConfiguration
from app.kpis import get_registry

# ── Priorities ────────────────────────────────────────────────────────────────

_PRIORITIES = [
    ("Critical", "#DC2626", 1),
    ("High", "#F59E0B", 2),
    ("Medium", "#FBBF24", 3),
    ("Low", "#10B981", 4),
]

# ── Zones, derived from the real camera-name vocabulary (offshore rig/vessel) ──
# (zone_name, [keywords to match against Camera.name, case-insensitive], default_priority_name)
_ZONE_RULES: list[tuple[str, list[str], str]] = [
    ("Wheel House", ["WHEEL HOUSE"], "High"),
    ("Helideck & Radar", ["HELIDECK", "RADAR"], "High"),
    ("Jack House", ["JACK HOUSE"], "High"),
    ("Lifeboat & Rescue Stations", ["LIFEBOAT", "RESCUE BOAT"], "High"),
    ("A Deck & Walkways", ["A DECK", "WALKWAY"], "Medium"),
    ("Main Deck", ["MAIN DECK", "MAIN"], "Medium"),
    ("Engine Room & ECR", ["ENGINE ROOM", "ECR", "VFD"], "Critical"),
    ("Laundry", ["LAUNDRY"], "Low"),
    ("Crane Area", ["CRANE"], "Critical"),
]
_FALLBACK_ZONE = ("General Area", "Medium")

# kpi registry name -> capabilities that make sense for that zone
_ZONE_KPI_DEFAULTS: dict[str, list[str]] = {
    "Wheel House": ["mobile_usage", "ppe"],
    "Helideck & Radar": ["people_count", "ppe"],
    "Jack House": ["ppe", "fire_smoke"],
    "Lifeboat & Rescue Stations": ["people_count", "ppe"],
    "A Deck & Walkways": ["ppe", "floating"],
    "Main Deck": ["ppe", "object_detection", "floating"],
    "Engine Room & ECR": ["fire_smoke", "ppe", "smoking"],
    "Laundry": ["fire_smoke"],
    "Crane Area": ["ppe", "falling_pose"],
    "General Area": ["ppe"],
}

# registry name -> the config.json key each detector's primary model lives
# under (see each KPI class's config block — some compound detectors like
# MobileUsageKPI/SmokingKPI have several *_model_path keys; the one picked
# here is that KPI's most identifying model, not necessarily the only one).
_KPI_MODEL_PATH_KEY: dict[str, tuple[str, str]] = {
    "fire_smoke": ("FireSmokeKPI", "model_path"),
    "ppe": ("PPEKPI", "model_path"),
    "falling_pose": ("FallingPoseKPI", "pose_model_path"),
    "floating": ("FloatingKPI", "model_path"),
    "speed_tracker": ("SpeedTrackerKPI", "model_path"),
    "smoking": ("SmokingKPI", "cigarette_model_path"),
    "mobile_usage": ("MobileUsageKPI", "phone_model_path"),
    "ANPR_LPR": ("AnprLprKPI", "model_path"),
    "object_detection": ("BoxCounterKPI", "model_path"),
    "density_occupancy": ("DensityOccupancyKPI", "model_path"),
    "people_count": ("PeopleCountKPI", "model_path"),
}

_SAMPLE_USERS = [
    # (full_name, username, email_local_part, phone, role_name)
    ("Ravi Operator", "ravi.operator", "ravi.operator", "+15550101001", "Operator"),
    ("Elena Guard", "elena.guard", "elena.guard", "+15550101002", "Zone Guard"),
]


class _Stub:
    """Placeholder for a would-be-created row in --dry-run mode, so
    downstream steps that reference e.g. zone.id can still run their full
    preview logic instead of silently skipping (dry_run never writes
    anything regardless — this only makes the printed preview trustworthy)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.id = None


def seed_priorities(session: Session, org: Organization, *, dry_run: bool) -> dict[str, Priority]:
    existing = {p.name: p for p in session.exec(select(Priority).where(Priority.org_id == org.id)).all()}
    for name, color, level in _PRIORITIES:
        if name in existing:
            continue
        print(f"  + priority {name} ({color}, level {level})")
        if dry_run:
            existing[name] = _Stub(name=name)
            continue
        p = Priority(name=name, color=color, level=level, org_id=org.id, created_by="seed_reference_data.py")
        session.add(p)
        session.flush()
        existing[name] = p
    return existing


def seed_zones(session: Session, org: Organization, *, dry_run: bool) -> dict[str, Zone]:
    existing = {z.name: z for z in session.exec(select(Zone).where(Zone.org_id == org.id)).all()}
    all_zone_names = [rule[0] for rule in _ZONE_RULES] + [_FALLBACK_ZONE[0]]
    for zone_name in all_zone_names:
        if zone_name in existing:
            continue
        print(f"  + zone {zone_name}")
        if dry_run:
            existing[zone_name] = _Stub(name=zone_name)
            continue
        z = Zone(name=zone_name, org_id=org.id, created_by="seed_reference_data.py")
        session.add(z)
        session.flush()
        existing[zone_name] = z
    return existing


def _zone_for_camera(name: str) -> tuple[str, str]:
    upper = name.upper()
    for zone_name, keywords, priority_name in _ZONE_RULES:
        if any(kw in upper for kw in keywords):
            return zone_name, priority_name
    return _FALLBACK_ZONE[0], _FALLBACK_ZONE[1]


def assign_camera_references(
    session: Session, org: Organization, zones: dict[str, Zone], priorities: dict[str, Priority],
    *, force: bool, dry_run: bool,
) -> int:
    registered = set(get_registry().keys())
    cameras = session.exec(select(Camera).where(Camera.org_id == org.id).order_by(Camera.camera_id)).all()
    updated = 0
    for cam in cameras:
        zone_name, priority_name = _zone_for_camera(cam.name)
        zone = zones.get(zone_name)
        priority = priorities.get(priority_name)
        kpi_defaults = [k for k in _ZONE_KPI_DEFAULTS.get(zone_name, []) if k in registered]

        needs_zone = force or cam.zone_id is None
        needs_priority = force or cam.priority_id is None
        needs_kpis = force or not cam.kpi_model_ids

        if not (needs_zone or needs_priority or needs_kpis):
            continue

        print(f"  ~ {cam.camera_id} ({cam.name}) -> zone={zone_name!r}, priority={priority_name!r}, kpi_model_ids={kpi_defaults}")
        updated += 1
        if dry_run:
            continue
        if needs_zone and zone:
            cam.zone_id = zone.id
        if needs_priority and priority:
            cam.priority_id = priority.id
        if needs_kpis:
            cam.kpi_model_ids = kpi_defaults
        session.add(cam)
    return updated


def seed_kpi_model_catalog(session: Session, org: Organization, *, dry_run: bool) -> dict[str, KpiModelCatalog]:
    """One catalog entry per registered KPI, using that detector's real
    model_path + confidence straight from config.json — not invented."""
    existing = {m.name: m for m in session.exec(select(KpiModelCatalog).where(KpiModelCatalog.org_id == org.id)).all()}
    by_kpi_name: dict[str, KpiModelCatalog] = {}
    registry = get_registry()

    for kpi_name, cls in registry.items():
        class_name, path_key = _KPI_MODEL_PATH_KEY.get(kpi_name, (cls.__name__, "model_path"))
        cfg = get_kpi_config(class_name)
        model_path = cfg.get(path_key)
        confidence = cfg.get("confidence", 0.5)
        if not model_path:
            continue

        catalog_name = f"{cls.display_name} Model"
        if catalog_name not in existing:
            print(f"  + kpi model {catalog_name} -> {model_path} (confidence {confidence})")
            if dry_run:
                existing[catalog_name] = _Stub(name=catalog_name)
            else:
                entry = KpiModelCatalog(
                    name=catalog_name, model_path=model_path, confidence_threshold=confidence,
                    org_id=org.id, created_by="seed_reference_data.py",
                )
                session.add(entry)
                session.flush()
                existing[catalog_name] = entry
        if catalog_name in existing:
            by_kpi_name[kpi_name] = existing[catalog_name]
    return by_kpi_name


def link_kpi_configuration_models(
    session: Session, org: Organization, model_by_kpi: dict[str, KpiModelCatalog], *, force: bool, dry_run: bool,
) -> int:
    configs = session.exec(select(KPIConfiguration).where(KPIConfiguration.org_id == org.id)).all()
    updated = 0
    for config in configs:
        if config.assigned_models and not force:
            continue
        model = model_by_kpi.get(config.kpi_name)
        if model is None:
            continue
        print(f"  ~ kpi_configuration[{config.kpi_name}].assigned_models -> [{model.id}]")
        updated += 1
        if not dry_run:
            config.assigned_models = [model.id]
            session.add(config)
    return updated


def seed_roles(session: Session, org: Organization, zones: dict[str, Zone], *, dry_run: bool) -> dict[str, Role]:
    existing = {r.name: r for r in session.exec(select(Role).where(Role.org_id == org.id)).all()}

    engine_room_zone = zones.get("Engine Room & ECR")
    to_create = [
        ("Operator", "Day-to-day camera/alert monitoring — no user/role/org administration.",
         {"dashboard": ["view"], "cameras": ["view", "edit"], "alerts": ["view"], "kpi_management": ["view"]},
         []),
        ("Zone Guard", "Restricted to the Engine Room & ECR zone only.",
         {"dashboard": ["view"], "cameras": ["view"], "alerts": ["view"]},
         [engine_room_zone.id] if engine_room_zone else []),
    ]
    for name, description, permissions, zone_ids in to_create:
        if name in existing:
            continue
        print(f"  + role {name} (zone_ids={zone_ids})")
        if dry_run:
            existing[name] = _Stub(name=name)
            continue
        role = Role(
            name=name, description=description, permissions=permissions,
            zone_ids=zone_ids, is_system=False, org_id=org.id,
        )
        session.add(role)
        session.flush()
        existing[name] = role
    return existing


def seed_users(session: Session, org: Organization, roles: dict[str, Role], *, dry_run: bool) -> int:
    created = 0
    for full_name, username, email_local, phone, role_name in _SAMPLE_USERS:
        if session.exec(select(User).where(User.username == username)).first() is not None:
            continue
        role = roles.get(role_name)
        if role is None:
            continue
        print(f"  + user {username} (role={role_name})")
        created += 1
        if dry_run:
            continue
        session.add(User(
            full_name=full_name,
            personal_email=f"{email_local}@example.com",
            login_email=f"{email_local}@example.com",
            phone=phone,
            username=username,
            password_hash=hash_password("Str0ng!Passw0rd"),
            org_id=org.id, role_id=role.id, status="active", verify_status=True,
            created_by="seed_reference_data.py",
        ))
    return created


def seed_email_logs(session: Session, org: Organization, *, dry_run: bool) -> int:
    """A plausible subset of the org's alerts get a notification-sent (or,
    occasionally, failed) log row — not literally every alert, since not
    every detection is email-worthy in practice."""
    already_logged = set(session.exec(select(EmailLog.alert_id)).all())
    cameras = session.exec(select(Camera).where(Camera.org_id == org.id)).all()
    camera_names = {c.camera_id: c.name for c in cameras}
    alerts = session.exec(select(Alert).where(Alert.camera_id.in_(camera_names.keys()))).all()

    created = 0
    for alert in alerts:
        if alert.id in already_logged:
            continue
        if alert.confidence < 0.75:  # only "notable" detections get emailed
            continue
        created += 1
        if dry_run:
            continue
        failed = random.random() < 0.08
        session.add(EmailLog(
            alert_id=alert.id, kpi_name=alert.kpi_name, alert_type=alert.alert_type,
            camera_id=alert.camera_id, camera_name=camera_names.get(alert.camera_id, alert.camera_id),
            subject=f"[Vision AI] {alert.alert_type.replace('_', ' ').title()} on {alert.camera_id}",
            recipients=["safety.officer@example.com"],
            status="failed" if failed else "sent",
            error="SMTP timeout" if failed else None,
            created_at=alert.created_at,
        ))
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--org-id", type=int, default=None, help="Seed into this organization (default: first org).")
    parser.add_argument("--force", action="store_true", help="Overwrite cameras'/kpi_configuration's existing references.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without writing anything.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible email_logs sampling.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    with Session(get_engine()) as session:
        if args.org_id is not None:
            org = session.get(Organization, args.org_id)
            if org is None:
                print(f"No organization with id={args.org_id}.")
                return
        else:
            org = session.exec(select(Organization).order_by(Organization.id)).first()
            if org is None:
                print("No organizations exist yet — nothing to seed into.")
                return

        print(f"Seeding reference data into organization {org.id} ({org.org_id}){' [DRY RUN]' if args.dry_run else ''}:")

        print("\nPriorities:")
        priorities = seed_priorities(session, org, dry_run=args.dry_run)

        print("\nZones:")
        zones = seed_zones(session, org, dry_run=args.dry_run)

        print("\nCamera zone_id/priority_id/kpi_model_ids:")
        cameras_updated = assign_camera_references(session, org, zones, priorities, force=args.force, dry_run=args.dry_run)

        print("\nKPI Model Catalog (detection models):")
        model_by_kpi = seed_kpi_model_catalog(session, org, dry_run=args.dry_run)

        print("\nKPI Management catalog -> assigned_models links:")
        kpi_config_updated = link_kpi_configuration_models(session, org, model_by_kpi, force=args.force, dry_run=args.dry_run)

        print("\nRoles:")
        roles = seed_roles(session, org, zones, dry_run=args.dry_run)

        print("\nUsers:")
        users_created = seed_users(session, org, roles, dry_run=args.dry_run)

        print("\nEmail logs (notification history for existing alerts):")
        email_logs_created = seed_email_logs(session, org, dry_run=args.dry_run)

        if not args.dry_run:
            session.commit()

        mode = "Would write" if args.dry_run else "Wrote/updated"
        print(
            f"\n{mode}: {len(priorities)} priorities, {len(zones)} zones, "
            f"{cameras_updated} camera(s) updated, {len(model_by_kpi)} kpi models, "
            f"{kpi_config_updated} kpi_configuration link(s), {len(roles)} roles, "
            f"{users_created} user(s), {email_logs_created} email_log(s)."
        )


if __name__ == "__main__":
    main()
