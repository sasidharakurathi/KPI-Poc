"""Canonical permission-matrix vocabulary.

Must exactly match the frontend's PERMISSION_MODULES / PERMISSION_ACTIONS
constants (vision-ai-frontend/src/types/domain.ts) — Role.permissions is
stored as Record<module, action[]>, e.g. {"cameras": ["view", "edit"]}.
"""

PERMISSION_MODULES: list[str] = [
    "dashboard",
    "cameras",
    "kpi_management",
    "alerts",
    "configuration",
    "roles",
    "users",
    "audit_log",
    "organization_settings",
]

PERMISSION_ACTIONS: list[str] = ["view", "create", "edit", "delete"]


def full_permission_matrix() -> dict[str, list[str]]:
    """Every module, every action — used for the built-in Owner role."""
    return {module: list(PERMISSION_ACTIONS) for module in PERMISSION_MODULES}


def is_valid_permission_matrix(value: object) -> bool:
    """True if `value` is a well-formed Record<module, action[]> using only
    known module/action names. Used to validate Role.permissions on write."""
    if not isinstance(value, dict):
        return False
    for module, actions in value.items():
        if module not in PERMISSION_MODULES:
            return False
        if not isinstance(actions, list) or not all(a in PERMISSION_ACTIONS for a in actions):
            return False
    return True
