"""Centralized configuration loaded from environment variables (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)

_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return default


@dataclass(frozen=True)
class RegisterSettings:
    unmasked_samples: int
    masked_samples: int
    min_confidence: float
    capture_delay: float
    camera_index: int
    use_gpu: bool


@dataclass(frozen=True)
class RecognitionSettings:
    camera_index: int
    min_confidence: float
    threshold: float
    det_size: int
    use_gpu: bool


register_settings = RegisterSettings(
    unmasked_samples=_get_int("FACE_REG_UNMASKED_SAMPLES", 20),
    masked_samples=_get_int("FACE_REG_MASKED_SAMPLES", 12),
    min_confidence=_get_float("FACE_REG_MIN_CONFIDENCE", 0.35),
    capture_delay=_get_float("FACE_REG_CAPTURE_DELAY", 0.45),
    camera_index=_get_int("FACE_REG_CAMERA_INDEX", 0),
    use_gpu=_get_bool("FACE_REG_USE_GPU", False),
)

recognition_settings = RecognitionSettings(
    camera_index=_get_int("FACE_RECOG_CAMERA_INDEX", 0),
    min_confidence=_get_float("FACE_RECOG_MIN_CONFIDENCE", 0.35),
    threshold=_get_float("FACE_RECOG_THRESHOLD", 0.32),
    det_size=_get_int("FACE_RECOG_DET_SIZE", 640),
    use_gpu=_get_bool("FACE_RECOG_USE_GPU", False),
)


__all__ = ["register_settings", "recognition_settings"]
