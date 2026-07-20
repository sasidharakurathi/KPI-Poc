"""Timezone catalog endpoint — Phase 1.

Implements:
  GET /api/timezones   List every enabled timezone (id, abbreviation, name, offsets)

Static, non-editable reference data — deliberately no POST/PUT/DELETE/PATCH.
Seeded once at startup (app/db/seed_data/timezones.py) from a fixed ~200-row
list transcribed from the org's reference schema script.

Deliberately public (no auth): this must be fetchable during org
registration, before any organization/user/token exists yet, so the
registration form can populate a "default timezone" dropdown. Every other
Configuration submodule requires auth; this one can't.

Models used: Timezone (app.db.models.domain_config)
Schemas: TimezoneResponse (app.schemas.timezone)
"""
from fastapi import APIRouter
from sqlmodel import select

from app.core.dependencies import DbSession
from app.db.models.domain_config import Timezone
from app.schemas.timezone import TimezoneResponse

router = APIRouter(prefix="/api/timezones", tags=["timezones"])


@router.get("", response_model=list[TimezoneResponse])
def list_timezones(session: DbSession) -> list[Timezone]:
    return session.exec(
        select(Timezone).where(Timezone.enabled == True).order_by(Timezone.abbreviation)  # noqa: E712
    ).all()
