import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlmodel import Session, select

from app.config_loader import get_kpi_config, get_kpi_param
from app.db.engine import get_engine
from app.db.models import Organization
from app.db.models.kpi_configuration import KPIConfiguration
from app.kpis import get_registry


_METADATA: dict[str, tuple[str, str]] = {
    "fire_smoke":      ("safety",     "Detects fire and smoke events in camera feeds."),
    "ppe":             ("safety",     "Detects personal protective equipment compliance (helmets, vests, etc.)."),
    "falling_pose":    ("safety",     "Detects a person falling or in a fallen pose."),
    "floating":        ("safety",     "Detects a floating object or person overboard in water areas."),
    "speed_tracker":   ("safety",     "Detects and tracks vehicle speed."),
    "smoking":         ("compliance", "Detects smoking activity in restricted zones."),
    "mobile_usage":    ("compliance", "Detects mobile phone usage during work activity."),
    "ANPR_LPR":        ("security",   "Automatic number plate recognition for vehicles."),
    "object_detection":("operations", "Detects and counts cartons/boxes in the frame."),
    "density_occupancy": ("operations", "Monitors zone density and occupancy levels."),
    "people_count":    ("operations", "Counts people present in the camera's field of view."),
}
_DEFAULT_METADATA = ("operations", "")


def seed(session: Session, org: Organization, *, force: bool, dry_run: bool) -> tuple[int, int, int]:
    """Returns (created, refreshed, skipped) counts for this org."""
    created = refreshed = skipped = 0

    for cls in get_registry().values():
        existing = session.exec(
            select(KPIConfiguration).where(
                KPIConfiguration.kpi_name == cls.name, KPIConfiguration.org_id == org.id,
            )
        ).first()

        parameters = get_kpi_config(cls.__name__)
        enabled = get_kpi_param(cls.__name__, "enabled", True)
        category, description = _METADATA.get(cls.name, _DEFAULT_METADATA)

        if existing is None:
            print(f"  [{org.org_id}] + {cls.name:<20} ({cls.display_name}) -> category={category}, enabled={enabled}")
            created += 1
            if not dry_run:
                session.add(KPIConfiguration(
                    kpi_name=cls.name, org_id=org.id,
                    description=description, category=category,
                    parameters=parameters, enable_status=enabled,
                    created_by="seed_kpi_catalog.py",
                ))
        elif force:
            print(f"  [{org.org_id}] ~ {cls.name:<20} refreshing parameters/enabled from config.json")
            refreshed += 1
            if not dry_run:
                existing.parameters = parameters
                existing.enable_status = enabled
                existing.updated_by = "seed_kpi_catalog.py"
                session.add(existing)
        else:
            skipped += 1

    return created, refreshed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--org-id", type=int, default=None, help="Seed only this organization (default: every org).")
    parser.add_argument("--force", action="store_true", help="Refresh parameters/enabled on existing rows from config.json.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without writing anything.")
    args = parser.parse_args()

    with Session(get_engine()) as session:
        if args.org_id is not None:
            org = session.get(Organization, args.org_id)
            if org is None:
                print(f"No organization with id={args.org_id}.")
                return
            orgs = [org]
        else:
            orgs = list(session.exec(select(Organization).order_by(Organization.id)).all())

        if not orgs:
            print("No organizations exist yet — nothing to seed.")
            return

        total_created = total_refreshed = total_skipped = 0
        for org in orgs:
            print(f"Organization {org.id} ({org.org_id}):")
            created, refreshed, skipped = seed(session, org, force=args.force, dry_run=args.dry_run)
            total_created += created
            total_refreshed += refreshed
            total_skipped += skipped

        if not args.dry_run:
            session.commit()

        mode = "DRY RUN — " if args.dry_run else ""
        print(f"\n{mode}{total_created} created, {total_refreshed} refreshed, {total_skipped} already present, "
              f"across {len(orgs)} organization(s).")


if __name__ == "__main__":
    main()
