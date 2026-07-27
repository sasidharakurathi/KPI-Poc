"""Seeds a KPIConfiguration catalog row for every registered detector
(app.kpis.registry) - one global row per KPI, shared across every
organization on this deployment (see app.db.models.kpi_configuration
.KPIConfiguration's docstring; there's a single real detection pipeline
behind each KPI regardless of tenant count).

Without this, GET /api/kpis/catalog is empty until someone manually PUTs
each of the 11 registered detectors one at a time - this does that in bulk,
using each detector's live config.json block as the starting `parameters`
value so the DB row matches what the pipeline is actually running today.

Idempotent: a KPI that already has a row is left untouched (use --force to
overwrite parameters/enabled from config.json anyway; category/description
are never overwritten by --force, since those may have been hand-edited via
the API).

Usage:
    python scripts/seed_kpi_catalog.py                # seed every missing KPI
    python scripts/seed_kpi_catalog.py --force         # refresh parameters/enabled from config.json
    python scripts/seed_kpi_catalog.py --dry-run        # show what would change, write nothing
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlmodel import Session, select

from app.config_loader import get_kpi_config, get_kpi_param
from app.db.engine import get_engine
from app.db.models.kpi_configuration import KPIConfiguration
from app.kpis import get_registry

# Sensible defaults for the 11 KPIs currently registered - category must be
# one of KPI_CATEGORIES (app.schemas.kpi). Anything registered later that
# isn't listed here falls back to ("operations", "") rather than failing.
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


def seed(session: Session, *, force: bool, dry_run: bool) -> tuple[int, int, int]:
    """Returns (created, refreshed, skipped) counts."""
    created = refreshed = skipped = 0

    for cls in get_registry().values():
        existing = session.exec(
            select(KPIConfiguration).where(KPIConfiguration.kpi_name == cls.name)
        ).first()

        parameters = get_kpi_config(cls.__name__)
        enabled = get_kpi_param(cls.__name__, "enabled", True)
        category, description = _METADATA.get(cls.name, _DEFAULT_METADATA)

        if existing is None:
            print(f"  + {cls.name:<20} ({cls.display_name}) -> category={category}, enabled={enabled}")
            created += 1
            if not dry_run:
                session.add(KPIConfiguration(
                    kpi_name=cls.name,
                    description=description, category=category,
                    parameters=parameters, enable_status=enabled,
                    created_by="seed_kpi_catalog.py",
                ))
        elif force:
            print(f"  ~ {cls.name:<20} refreshing parameters/enabled from config.json")
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
    parser.add_argument("--force", action="store_true", help="Refresh parameters/enabled on existing rows from config.json.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without writing anything.")
    args = parser.parse_args()

    with Session(get_engine()) as session:
        created, refreshed, skipped = seed(session, force=args.force, dry_run=args.dry_run)

        if not args.dry_run:
            session.commit()

        mode = "DRY RUN - " if args.dry_run else ""
        print(f"\n{mode}{created} created, {refreshed} refreshed, {skipped} already present.")


if __name__ == "__main__":
    main()
