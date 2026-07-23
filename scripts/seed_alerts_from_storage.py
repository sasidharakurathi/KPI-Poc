"""Clears Alert/AlertFrame/Job rows and repopulates them from the real frame
images already sitting on disk under storage/alerts/ — dummy DB rows backed
by real files, so GET /api/alerts, frame/labeled-frame downloads, exports,
and the dashboard all have something genuine-looking to show.

storage/alerts/ layout (produced by the real detection pipeline —
app.kpis.base's alert-saving logic):

    storage/alerts/<job_id>/<kpi_name>/<seq>/
        00_frame<N>.jpg .. 07_frame<N>.jpg   8 raw frames: ALERT_WINDOW_BEFORE
                                              (4) + anchor + ALERT_WINDOW_AFTER (3)
        labeled_frame<N>.jpg                 annotated copy of the anchor frame
                                              (position 4) — only present when
                                              the job ran in developer mode

<seq> is a running per-job save counter, not a frame index — the real anchor
frame_idx is the number in the position-4 filename (cross-checked against the
labeled file's frame number when present). This script derives every Alert/
AlertFrame field the same way the real pipeline would have, then assigns each
<job_id> one of the target org's real cameras (round-robin) and a
randomized-but-plausible timestamp, since the original camera/timing metadata
isn't recoverable from the files alone.

THIS IS DESTRUCTIVE, but scoped to the target org: every existing jobs/
alerts/alert_frames row tied to the target org's cameras is deleted first
(printed with counts before anything happens) — other organizations' alert
history on the same deployment is untouched. Camera/Zone/Priority/User/etc.
tables are untouched everywhere.

Timestamps span multiple calendar years, not just the recent past: the full
211-alert dataset is replicated once per target year (this year, and
years_back-1 years before it — see --years), each replica getting its own
job_id (suffixed with the year) and a timestamp randomized within that one
year. This is what gives dashboard.get_cameras' alerts_by_year (grouped by
year, summed across every KPI for that camera) something real to show.

Usage:
    python scripts/seed_alerts_from_storage.py                # first org, real run, 3 years back
    python scripts/seed_alerts_from_storage.py --org-id 2
    python scripts/seed_alerts_from_storage.py --years 5       # spread across 5 calendar years instead
    python scripts/seed_alerts_from_storage.py --dry-run       # preview only, no DB writes
"""
import argparse
import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlmodel import Session, select

from app.config import BASE_DIR, settings
from app.db.engine import get_engine
from app.db.models import Alert, AlertFrame, Camera, Job, Organization

ANCHOR_POSITION = 4  # matches settings.ALERT_WINDOW_BEFORE

_POSITION_RE = re.compile(r"^(\d+)_frame(\d+)\.jpg$")
_LABELED_RE = re.compile(r"^labeled_frame(\d+)\.jpg$")

_ALERT_TYPE_BY_KPI: dict[str, str] = {
    "fire_smoke": "fire_detected",
    "ppe": "ppe_violation",
    "falling_pose": "fall_detected",
    "floating": "person_overboard",
    "speed_tracker": "overspeed_detected",
    "smoking": "smoking_detected",
    "mobile_usage": "phone_usage_confirmed",
    "ANPR_LPR": "plate_recognized",
    "object_detection": "object_detected",
    "density_occupancy": "occupancy_threshold_exceeded",
    "people_count": "people_count_alert",
}


class _ParsedEvent:
    def __init__(self, job_id: str, kpi_name: str, seq_dir: Path):
        self.job_id = job_id
        self.kpi_name = kpi_name
        self.seq_dir = seq_dir
        self.frames: list[tuple[int, int, Path]] = []  # (position, frame_number, path)
        self.labeled: dict[int, Path] = {}  # frame_number -> labeled file path

    def add(self, path: Path) -> None:
        if (m := _POSITION_RE.match(path.name)):
            self.frames.append((int(m.group(1)), int(m.group(2)), path))
        elif (m := _LABELED_RE.match(path.name)):
            self.labeled[int(m.group(1))] = path

    def anchor_frame_idx(self) -> int | None:
        by_position = {pos: fnum for pos, fnum, _ in self.frames}
        if ANCHOR_POSITION in by_position:
            return by_position[ANCHOR_POSITION]
        return next(iter(self.labeled), None)


def scan_storage(alerts_dir: Path) -> list[_ParsedEvent]:
    events: list[_ParsedEvent] = []
    if not alerts_dir.exists():
        return events
    for job_dir in sorted(p for p in alerts_dir.iterdir() if p.is_dir()):
        for kpi_dir in sorted(p for p in job_dir.iterdir() if p.is_dir()):
            for seq_dir in sorted(p for p in kpi_dir.iterdir() if p.is_dir()):
                event = _ParsedEvent(job_dir.name, kpi_dir.name, seq_dir)
                for f in seq_dir.iterdir():
                    if f.is_file():
                        event.add(f)
                if event.frames:
                    events.append(event)
    return events


def _rel(path: Path) -> str:
    # settings.ALERTS_DIR (and therefore every path derived from it while
    # scanning) may be relative — .env overrides it to "storage/alerts"
    # rather than the config.py default's already-absolute BASE_DIR/storage/
    # alerts — so both sides are normalized to absolute before comparing.
    return str(path.resolve().relative_to(BASE_DIR.resolve()))


def _org_scoped_rows(session: Session, org: Organization) -> tuple[list[Alert], list[Job]]:
    """Every Alert/Job tied to this org's cameras — via Alert.camera_id
    directly, or Job.camera_id (covers a job row with no alerts, e.g. one
    that failed before producing any detection)."""
    camera_ids = set(session.exec(select(Camera.camera_id).where(Camera.org_id == org.id)).all())
    if not camera_ids:
        return [], []
    alerts = session.exec(select(Alert).where(Alert.camera_id.in_(camera_ids))).all()
    job_ids = {a.job_id for a in alerts if a.job_id} | set(
        session.exec(select(Job.job_id).where(Job.camera_id.in_(camera_ids))).all()
    )
    jobs = session.exec(select(Job).where(Job.job_id.in_(job_ids))).all() if job_ids else []
    return list(alerts), list(jobs)


def clear_existing(session: Session, org: Organization) -> tuple[int, int, int]:
    alerts, jobs = _org_scoped_rows(session, org)
    alert_ids = {a.id for a in alerts}
    frames = session.exec(select(AlertFrame).where(AlertFrame.alert_id.in_(alert_ids))).all() if alert_ids else []

    for row in frames:
        session.delete(row)
    for row in alerts:
        session.delete(row)
    for row in jobs:
        session.delete(row)
    session.commit()
    return len(frames), len(alerts), len(jobs)


def _random_timestamp_in_year(year: int, now: datetime) -> datetime:
    """A random moment within `year` — capped at `now` when year is the
    current year, so we never generate a future timestamp."""
    start = datetime(year, 1, 1)
    end = min(datetime(year, 12, 31, 23, 59, 59), now) if year == now.year else datetime(year, 12, 31, 23, 59, 59)
    span_seconds = max((end - start).total_seconds(), 1)
    return start + timedelta(seconds=random.uniform(0, span_seconds))


def seed(
    session: Session, org: Organization, events: list[_ParsedEvent], *, years_back: int, dry_run: bool,
) -> tuple[int, int, int]:
    """Replicates the full real-file-backed dataset once per target year
    (this year, and years_back-1 years before it) — same real frame images
    every time (paths are just references, not consumed per-alert), but each
    replica gets its own job_id (suffixed with the year) and a timestamp
    randomized within that specific calendar year. This is what actually
    produces multi-year data for dashboard.get_cameras' alerts_by_year,
    rather than one random timestamp per job spread thin across years."""
    cameras = list(session.exec(select(Camera).where(Camera.org_id == org.id).order_by(Camera.camera_id)).all())
    if not cameras:
        print(f"Organization {org.id} ({org.org_id}) has no cameras — nothing to assign alerts to.")
        return 0, 0, 0

    by_job: dict[str, list[_ParsedEvent]] = {}
    for event in events:
        by_job.setdefault(event.job_id, []).append(event)

    now = datetime.utcnow()
    target_years = [now.year - offset for offset in range(years_back)]
    jobs_written = alerts_written = frames_written = 0

    for year in target_years:
        print(f"  -- year {year} --")
        for i, (job_id, job_events) in enumerate(sorted(by_job.items())):
            camera = cameras[i % len(cameras)]
            year_job_id = f"{job_id}-y{year}"
            job_created_at = _random_timestamp_in_year(year, now)
            kpi_names = sorted({e.kpi_name for e in job_events})

            print(f"  job {job_id[:8]}...-y{year} -> {camera.camera_id} ({len(job_events)} alert(s): {', '.join(kpi_names)})")
            jobs_written += 1
            if not dry_run:
                session.add(Job(
                    job_id=year_job_id, filename=f"{job_id}.mp4", video_path=f"storage/uploads/{job_id}.mp4",
                    camera_id=camera.camera_id, camera_name=camera.name, status="completed",
                    kpis_running=kpi_names, created_at=job_created_at,
                    completed_at=job_created_at + timedelta(minutes=random.uniform(1, 5)),
                ))

            for j, event in enumerate(sorted(job_events, key=lambda e: (e.kpi_name, e.seq_dir.name))):
                frame_idx = event.anchor_frame_idx()
                if frame_idx is None:
                    continue
                alert_created_at = job_created_at + timedelta(seconds=j * random.uniform(5, 20))
                confidence = round(random.uniform(0.55, 0.98), 3)
                alert_type = _ALERT_TYPE_BY_KPI.get(event.kpi_name, f"{event.kpi_name}_detected")

                alerts_written += 1
                if dry_run:
                    frames_written += len(event.frames)
                    continue

                alert = Alert(
                    job_id=year_job_id, camera_id=camera.camera_id, org_id=camera.org_id, kpi_name=event.kpi_name,
                    alert_type=alert_type, frame_idx=frame_idx, confidence=confidence,
                    created_at=alert_created_at,
                )
                session.add(alert)
                session.flush()  # need alert.id for the frame rows below

                for position, frame_number, path in sorted(event.frames):
                    labeled = event.labeled.get(frame_number)
                    session.add(AlertFrame(
                        alert_id=alert.id, position=position, frame_idx=frame_number,
                        path=_rel(path), labeled_path=_rel(labeled) if labeled else None,
                    ))
                    frames_written += 1

    if not dry_run:
        session.commit()

    return jobs_written, alerts_written, frames_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--org-id", type=int, default=None, help="Assign alerts to this org's cameras (default: first org).")
    parser.add_argument("--years", type=int, default=3, help="Replicate the dataset across this many calendar years, ending this year (default: 3).")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without touching the database.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible dummy timestamps/confidences.")
    args = parser.parse_args()

    if args.years < 1:
        print("--years must be at least 1.")
        return

    if args.seed is not None:
        random.seed(args.seed)

    events = scan_storage(settings.ALERTS_DIR)
    print(f"Found {len(events)} alert-event director{'y' if len(events) == 1 else 'ies'} under {settings.ALERTS_DIR}")
    if not events:
        print("Nothing to seed.")
        return

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

        old_alerts, old_jobs = _org_scoped_rows(session, org)
        old_alert_ids = {a.id for a in old_alerts}
        old_frames = session.exec(select(AlertFrame).where(AlertFrame.alert_id.in_(old_alert_ids))).all() if old_alert_ids else []
        print(
            f"Existing rows for organization {org.id} ({org.org_id}) that will be cleared: "
            f"{len(old_jobs)} jobs, {len(old_alerts)} alerts, {len(old_frames)} alert_frames. "
            f"(Other organizations' rows are untouched.)"
        )

        if args.dry_run:
            print("DRY RUN — not clearing or writing anything.")
        else:
            cleared_frames, cleared_alerts, cleared_jobs = clear_existing(session, org)
            print(f"Cleared {cleared_jobs} jobs, {cleared_alerts} alerts, {cleared_frames} alert_frames.")

        print(f"Seeding into organization {org.id} ({org.org_id}), across {args.years} year(s):")
        jobs_written, alerts_written, frames_written = seed(
            session, org, events, years_back=args.years, dry_run=args.dry_run,
        )

        mode = "DRY RUN — would write" if args.dry_run else "Wrote"
        print(f"\n{mode} {jobs_written} jobs, {alerts_written} alerts, {frames_written} alert_frames.")


if __name__ == "__main__":
    main()
