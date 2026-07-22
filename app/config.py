from pathlib import Path
from typing import Optional
import torch
from pydantic import field_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class Settings(BaseSettings):
    UPLOAD_DIR: Path = BASE_DIR / "storage" / "uploads"
    ALERTS_DIR: Path = BASE_DIR / "storage" / "alerts"
    RECORDINGS_DIR: Path = BASE_DIR / "storage" / "recordings"
    LOGOS_DIR: Path = BASE_DIR / "storage" / "logos"
    LOGO_MAX_SIZE_BYTES: int = 2 * 1024 * 1024  # 2 MB
    DATABASE_URL: str = f"sqlite:///{(BASE_DIR / 'storage' / 'app.db').as_posix()}"
    DEVICE: str = _default_device()
    USE_HALF: bool = True
    DEV_MODE: bool = False
    ALERT_WINDOW_BEFORE: int = 4
    ALERT_WINDOW_AFTER: int = 3
    MAX_WORKERS: int = 4

    # IP camera continuous recording
    STREAM_CLIP_SECONDS: int = 120
    STREAM_RECONNECT_DELAY: float = 5.0
    STREAM_FRAME_THINNING_ENABLED: bool = True
    FRAME_THINNING_TARGET_FPS: float = 12.0

    FIRE_SMOKE_MODEL_PATH: Optional[Path] = None

    # Symmetric key for encrypting the SMTP password at rest (see .env.example)
    EMAIL_ENCRYPTION_KEY: Optional[str] = None

    # Base URL this backend is reachable at — used to build links in outgoing
    # account emails (activation, etc.). Not the frontend's URL.
    PUBLIC_BASE_URL: str = "http://localhost:8000"

    # ── Bootstrap SMTP defaults ─────────────────────────────────────────────
    # Used exactly once per organization: register_organization() copies these
    # into that org's first EmailServer row (is_default=True) so there's
    # something to send the very first activation email through, before any
    # admin has had a chance to configure one via Configuration > Email
    # Servers. If any required field here is unset, no row is seeded and
    # registration proceeds anyway (see app/services/email_service.py) — this
    # is a deployment-level convenience, not a hard requirement to sign up.
    DEFAULT_SMTP_HOST: Optional[str] = None
    DEFAULT_SMTP_PORT: int = 587
    DEFAULT_SMTP_USERNAME: Optional[str] = None
    DEFAULT_SMTP_PASSWORD: Optional[str] = None
    DEFAULT_SMTP_FROM_ADDRESS: Optional[str] = None
    DEFAULT_SMTP_FROM_NAME: str = "Vision AI Alerts"
    DEFAULT_SMTP_USE_TLS: bool = True

    # ── JWT / Auth ──────────────────────────────────────────────────────────
    JWT_AUTH_ENABLED: bool = False
    JWT_SECRET_KEY: Optional[str] = None
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 15              # short-lived access token (PRD: 15 min)
    JWT_REFRESH_EXPIRE_DAYS: int = 7          # server-tracked refresh token (PRD: 7 days)
    JWT_ISSUER: str = "vision-ai"

    # ── DB Migrations ───────────────────────────────────────────────────────
    # Set True to apply schema migrations on startup. Keep False in production
    # until you are ready to run them; then flip to True, restart once, flip back.
    MIGRATION_ENABLED: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @field_validator("FIRE_SMOKE_MODEL_PATH", mode="before")
    @classmethod
    def resolve_path(cls, v: Optional[str]) -> Optional[Path]:
        if v is None:
            return None
        p = Path(v)
        return p if p.is_absolute() else BASE_DIR / p


settings = Settings()
