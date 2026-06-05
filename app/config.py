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
    PROCESSED_DIR: Path = BASE_DIR / "storage" / "processed"

    DEVICE: str = _default_device()

    # Use FP16 on GPU for ~2× faster inference; ignored on CPU.
    USE_HALF: bool = True

    # Model paths – set these in .env (relative paths resolved from project root)
    FIRE_SMOKE_MODEL_PATH: Optional[Path] = None
    MOBILE_PERSON_MODEL_PATH: Optional[Path] = None
    MOBILE_PHONE_MODEL_PATH: Optional[Path] = None

    # Maximum threads used to run KPIs in parallel.
    MAX_WORKERS: int = 4

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @field_validator(
        "FIRE_SMOKE_MODEL_PATH", "MOBILE_PERSON_MODEL_PATH", "MOBILE_PHONE_MODEL_PATH",
        mode="before",
    )
    @classmethod
    def resolve_path(cls, v: Optional[str]) -> Optional[Path]:
        if v is None:
            return None
        p = Path(v)
        return p if p.is_absolute() else BASE_DIR / p


settings = Settings()
