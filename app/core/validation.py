"""Shared field-level validators used across multiple schema modules."""
import re

_PASSWORD_RULE = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^\w\s]).{8,128}$")


def validate_password_strength(value: str) -> str:
    """PRD: at least 8 characters, one uppercase, one lowercase, one number,
    one symbol. Used by Phase 0 (register/reset/change-password) and Phase 7
    (create-user/reset-password)."""
    if not _PASSWORD_RULE.match(value):
        raise ValueError(
            "Password must be at least 8 characters and include an uppercase "
            "letter, a lowercase letter, a number, and a symbol."
        )
    return value


_USERNAME_RULE = re.compile(r"^[a-zA-Z0-9._-]{3,32}$")


def validate_username(value: str) -> str:
    if not _USERNAME_RULE.match(value):
        raise ValueError(
            "Username must be 3-32 characters (letters, numbers, dots, dashes, underscores only)."
        )
    return value


_PHONE_RULE = re.compile(r"^\+[1-9]\d{7,14}$")


def validate_phone(value: str) -> str:
    if not _PHONE_RULE.match(value):
        raise ValueError("Enter a valid phone number in international format, e.g. +919876500001.")
    return value


def validate_non_blank(value: str, field_label: str = "This field") -> str:
    """Pydantic's min_length counts raw characters, so "  " (two spaces)
    passes a min_length=2 check and then gets silently stripped to "" by
    whichever service function calls .strip() on it - e.g. a role or
    organization ending up with an empty-string name. Reject whitespace-only
    input at the schema boundary instead. Returns the stripped value."""
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_label} cannot be blank.")
    return stripped
