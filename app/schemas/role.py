"""Role & permissions schemas — Phase 6.

Field names/shape match the frontend's real contract (`vision-ai-frontend/
src/types/domain.ts` Role/PermissionMatrix, `src/api/roles.ts` RoleInput) —
not the PRD-literal draft this file originally had. Notably:
  - ids are surfaced as strings (the DB uses int PKs internally; app.services
    .role_service converts at the boundary).
  - `permissions` is Record<module, action[]>, validated against
    app.core.permissions' canonical vocabulary.
  - `zone_ids` and `default_email_server_id` are also string-typed at the API
    boundary, converted to/from int for storage.
"""
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.core.permissions import is_valid_permission_matrix
from app.core.validation import validate_non_blank


class RoleInput(BaseModel):
    """Body for both POST /api/roles and PUT /api/roles/{id}."""

    name: str = Field(min_length=2, max_length=60)
    description: str = Field(default="", max_length=250)
    permissions: dict[str, list[str]]
    default_email_server_id: Optional[str] = None
    zone_ids: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_non_blank(cls, v: str) -> str:
        return validate_non_blank(v, "name")

    @field_validator("permissions")
    @classmethod
    def _valid_matrix(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        if not is_valid_permission_matrix(v):
            raise ValueError(
                "permissions must map known module names to known action names "
                "(see app.core.permissions.PERMISSION_MODULES/PERMISSION_ACTIONS)."
            )
        if not any(actions for actions in v.values()):
            raise ValueError("At least one module must have at least one permission enabled.")
        return v

    @field_validator("zone_ids")
    @classmethod
    def _dedupe_zone_ids(cls, v: list[str]) -> list[str]:
        """Silently collapse duplicates rather than rejecting them — the
        frontend builds this list from a multi-select and a stray duplicate
        shouldn't be a hard validation error."""
        seen: set[str] = set()
        deduped: list[str] = []
        for zid in v:
            if zid not in seen:
                seen.add(zid)
                deduped.append(zid)
        return deduped


class RoleResponse(BaseModel):
    id: str
    name: str
    description: str
    permissions: dict[str, list[str]]
    default_email_server_id: Optional[str] = None
    zone_ids: list[str]
    is_system: bool
    created_at: str
