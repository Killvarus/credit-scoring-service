"""Настройки из переменных окружения (12-factor): один образ — разные окружения."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_DIR", ROOT / "artifacts")))
    drift_window: int = field(default_factory=lambda: int(os.getenv("DRIFT_WINDOW", "1000")))
    drift_min_size: int = field(default_factory=lambda: int(os.getenv("DRIFT_MIN_SIZE", "100")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
