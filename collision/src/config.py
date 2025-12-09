
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


def _get_str(name: str, default: Optional[str]) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    return value


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
    video_path: Optional[str]
    video_prompt: bool


@dataclass(frozen=True)
class CollisionSettings:
    camera1_index: int
    camera2_index: int
    iou_threshold: float
    distance_threshold: float
    min_confidence: float
    recognition_threshold: float
    det_size: int
    use_gpu: bool
    enable_audio: bool
    min_risk_for_alert: str
    require_both_cameras: bool


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
    video_path=_get_str("FACE_RECOG_VIDEO_PATH", None),
    video_prompt=_get_bool("FACE_RECOG_VIDEO_PROMPT", False),
)

collision_settings = CollisionSettings(
    camera1_index=_get_int("COLLISION_CAMERA1_INDEX", 0),
    camera2_index=_get_int("COLLISION_CAMERA2_INDEX", 1),
    iou_threshold=_get_float("COLLISION_IOU_THRESHOLD", 0.1),
    distance_threshold=_get_float("COLLISION_DISTANCE_THRESHOLD", 200.0),
    min_confidence=_get_float("COLLISION_MIN_CONFIDENCE", 0.35),
    recognition_threshold=_get_float("COLLISION_RECOGNITION_THRESHOLD", 0.32),
    det_size=_get_int("COLLISION_DET_SIZE", 640),
    use_gpu=_get_bool("COLLISION_USE_GPU", False),
    enable_audio=_get_bool("COLLISION_ENABLE_AUDIO", False),
    min_risk_for_alert=_get_str("COLLISION_MIN_RISK_FOR_ALERT", "MEDIUM") or "MEDIUM",
    require_both_cameras=_get_bool("COLLISION_REQUIRE_BOTH_CAMERAS", True),
)


__all__ = ["register_settings", "recognition_settings"]
