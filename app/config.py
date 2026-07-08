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
    STREAM_FRAME_THINNING_ENABLED: bool = True   # keep every other frame + halve fps to cut processing time

    FIRE_SMOKE_MODEL_PATH: Optional[Path] = None

    # Symmetric key for encrypting the SMTP password at rest (see .env.example)
    EMAIL_ENCRYPTION_KEY: Optional[str] = None

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @field_validator("FIRE_SMOKE_MODEL_PATH", mode="before")
    @classmethod
    def resolve_path(cls, v: Optional[str]) -> Optional[Path]:
        if v is None:
            return None
        p = Path(v)
        return p if p.is_absolute() else BASE_DIR / p


settings = Settings()
