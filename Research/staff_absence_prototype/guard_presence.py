"""Guard Presence (KPI #19) — SWAP SEAM.

Per KPI_Pipeline_Architecture.md, Guard Presence is spec'd as:
    OBJ(person) -> ZONE -> SCHED
i.e. "is a person present in this zone, during the scheduled window" — which is exactly what
this placeholder computes from the confirmed-inside occupancy count that zone_occupancy.py
already produces. Nothing here is a hack; it's a spec-complete v1.

The teammate's real Guard Presence model will go further than "any person" — most likely a
uniform/role classifier or an enrolled-guard ReID match (the `UNI` primitive from the master
KPI doc), so that a visitor or cleaner walking through the post doesn't read as "guard present".
When that model is ready, swap the body of detect_guard_presence() (or replace it with a class
carrying model state) — absence.py only depends on this function's signature, so nothing
downstream needs to change.

KNOWN LIMITATION (until the swap): any tracked person inside the zone — guard, visitor, or
passer-by — counts as "present" and resets the absence timer. This can mask a real staff
absence if someone else happens to walk through the post.
"""


def detect_guard_presence(occupancy: int) -> bool:
    """occupancy: confirmed-inside person count from zone_occupancy.OccupancyZone.update().
    Returns True if a guard is considered present at the post this frame."""
    return occupancy > 0
