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

THIS IS DESTRUCTIVE: every existing row in jobs/alerts/alert_frames is
deleted first (see module docstring's "Clears" — printed with counts before
anything happens). Camera/Zone/Priority/User/etc. tables are untouched.

Usage:
    python scripts/seed_alerts_from_storage.py                # first org, real run
    python scripts/seed_alerts_from_storage.py --org-id 2
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


def clear_existing(session: Session) -> tuple[int, int, int]:
    frame_count = session.exec(select(AlertFrame)).all()
    alert_count = session.exec(select(Alert)).all()
    job_count = session.exec(select(Job)).all()
    for row in frame_count:
        session.delete(row)
    for row in alert_count:
        session.delete(row)
    for row in job_count:
        session.delete(row)
    session.commit()
    return len(frame_count), len(alert_count), len(job_count)


def seed(session: Session, org: Organization, events: list[_ParsedEvent], *, dry_run: bool) -> tuple[int, int, int]:
    cameras = list(session.exec(select(Camera).where(Camera.org_id == org.id).order_by(Camera.camera_id)).all())
    if not cameras:
        print(f"Organization {org.id} ({org.org_id}) has no cameras — nothing to assign alerts to.")
        return 0, 0, 0

    by_job: dict[str, list[_ParsedEvent]] = {}
    for event in events:
        by_job.setdefault(event.job_id, []).append(event)

    now = datetime.utcnow()
    jobs_written = alerts_written = frames_written = 0

    for i, (job_id, job_events) in enumerate(sorted(by_job.items())):
        camera = cameras[i % len(cameras)]
        job_created_at = now - timedelta(days=random.uniform(0, 30), hours=random.uniform(0, 23))
        kpi_names = sorted({e.kpi_name for e in job_events})

        print(f"  job {job_id[:8]}... -> {camera.camera_id} ({len(job_events)} alert(s): {', '.join(kpi_names)})")
        jobs_written += 1
        if not dry_run:
            session.add(Job(
                job_id=job_id, filename=f"{job_id}.mp4", video_path=f"storage/uploads/{job_id}.mp4",
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
                job_id=job_id, camera_id=camera.camera_id, kpi_name=event.kpi_name,
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
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without touching the database.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible dummy timestamps/confidences.")
    args = parser.parse_args()

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

        old_frames, old_alerts, old_jobs = len(session.exec(select(AlertFrame)).all()), \
            len(session.exec(select(Alert)).all()), len(session.exec(select(Job)).all())
        print(f"Existing rows that will be cleared: {old_jobs} jobs, {old_alerts} alerts, {old_frames} alert_frames.")

        if args.dry_run:
            print("DRY RUN — not clearing or writing anything.")
        else:
            cleared_frames, cleared_alerts, cleared_jobs = clear_existing(session)
            print(f"Cleared {cleared_jobs} jobs, {cleared_alerts} alerts, {cleared_frames} alert_frames.")

        print(f"Seeding into organization {org.id} ({org.org_id}):")
        jobs_written, alerts_written, frames_written = seed(session, org, events, dry_run=args.dry_run)

        mode = "DRY RUN — would write" if args.dry_run else "Wrote"
        print(f"\n{mode} {jobs_written} jobs, {alerts_written} alerts, {frames_written} alert_frames.")


if __name__ == "__main__":
    main()
