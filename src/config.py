"""Centralized configuration loaded from environment variables (.env)."""
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
    total_samples: int
    min_confidence: float
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
    enable_reid: bool
    reid_detector_conf: float
    reid_model_path: Optional[str]
    reid_embedder_gpu: bool
    reid_nms_iou: float
    reid_box_shrink: float


@dataclass(frozen=True)
class DualViewSettings:
    front_camera_index: int
    front_video_path: Optional[str]
    front_video_prompt: bool
    side_camera_index: int
    side_video_path: Optional[str]
    side_video_prompt: bool


@dataclass(frozen=True)
class ContactSettings:
    base_rate: float
    event_penalty: float
    mask_effect: float
    mask_decay_seconds: float
    overlap_threshold: float
    log_dir: Path
    pair_sync_window: float
    require_both_cameras: bool


@dataclass(frozen=True)
class CollisionSettings:
    iou_threshold: float
    distance_threshold: float
    distance_meters_threshold: float  # Real-world distance threshold in meters
    require_both_cameras: bool
    min_risk_for_alert: float
    alert_duration_seconds: float
    alert_cooldown_seconds: float
    enable_audio: bool
    alert_log_dir: Path

register_settings = RegisterSettings(
    total_samples=_get_int("FACE_REG_TOTAL_SAMPLES", 50),
    min_confidence=_get_float("FACE_REG_MIN_CONFIDENCE", 0.35),
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
    enable_reid=_get_bool("FACE_RECOG_ENABLE_REID", False),
    reid_detector_conf=_get_float("FACE_REID_DET_CONF", 0.35),
    reid_model_path=_get_str("FACE_REID_DETECTOR", "yolov8n.pt"),
    reid_embedder_gpu=_get_bool("FACE_REID_EMBEDDER_GPU", False),
    reid_nms_iou=_get_float("FACE_REID_NMS_IOU", 0.35),
    reid_box_shrink=_get_float("FACE_REID_BOX_SHRINK", 0.08),
)


dual_view_settings = DualViewSettings(
    front_camera_index=_get_int("FRONT_CAMERA_INDEX", 0),
    front_video_path=_get_str("FRONT_VIDEO_PATH", None),
    front_video_prompt=_get_bool("FRONT_VIDEO_PROMPT", False),
    side_camera_index=_get_int("SIDE_CAMERA_INDEX", 1),
    side_video_path=_get_str("SIDE_VIDEO_PATH", None),
    side_video_prompt=_get_bool("SIDE_VIDEO_PROMPT", False),
)


def _resolve_log_dir() -> Path:
    raw = _get_str("CONTACT_LOG_DIR", "Contact_Details") or "Contact_Details"
    path = Path(raw)
    if not path.is_absolute():
        return ROOT_DIR / path
    return path


def _resolve_alert_dir() -> Path:
    raw = _get_str("COLLISION_ALERT_LOG_DIR", "data/alerts") or "data/alerts"
    path = Path(raw)
    if not path.is_absolute():
        return ROOT_DIR / path
    return path


contact_settings = ContactSettings(
    base_rate=_get_float("CONTACT_BASE_RATE", 0.02),
    event_penalty=_get_float("CONTACT_EVENT_PENALTY", 0.05),
    mask_effect=_get_float("CONTACT_MASK_EFFECT", 0.5),
    mask_decay_seconds=_get_float("CONTACT_MASK_DECAY_SECONDS", 8.0),
    overlap_threshold=_get_float("CONTACT_OVERLAP_THRESHOLD", 0.18),
    log_dir=_resolve_log_dir(),
    pair_sync_window=_get_float("CONTACT_SYNC_WINDOW", 0.5),
    require_both_cameras=_get_bool("CONTACT_REQUIRE_BOTH_CAMERAS", False),
)


collision_settings = CollisionSettings(
    iou_threshold=_get_float("COLLISION_IOU_THRESHOLD", 0.1),
    distance_threshold=_get_float("COLLISION_DISTANCE_THRESHOLD", 200.0),
    distance_meters_threshold=_get_float("COLLISION_DISTANCE_METERS_THRESHOLD", 1.5),  # 1.5 meters default
    require_both_cameras=_get_bool("COLLISION_REQUIRE_BOTH_CAMERAS", True),
    min_risk_for_alert=_get_float("COLLISION_MIN_RISK_FOR_ALERT", 0.4),
    alert_duration_seconds=_get_float("COLLISION_ALERT_DURATION", 10.0),
    alert_cooldown_seconds=_get_float("COLLISION_ALERT_COOLDOWN", 12.0),
    enable_audio=_get_bool("COLLISION_ENABLE_AUDIO", False),
    alert_log_dir=_resolve_alert_dir(),
)

# Unknown person registration settings
UNKNOWN_REGISTER_MAX_IMAGES = _get_int("UNKNOWN_REGISTER_MAX_IMAGES", 50)
MDR_ALERT_THRESHOLD_SECONDS = _get_float("MDR_ALERT_THRESHOLD_SECONDS", 300.0)  # 5 minutes for MDR alerts
CONTACT_LOG_MIN_DURATION_SECONDS = _get_float("CONTACT_LOG_MIN_DURATION_SECONDS", 3.0)  # 3 seconds for logging contacts

# ============================================
# MDR Risk Calculation Settings
# Formula: R = (T * P * V) / D^2
# ============================================

# Pathogen Factors (P) - Higher values = more dangerous pathogen
MDR_PATHOGEN_FACTORS = {
    "MRSA": _get_float("MDR_PATHOGEN_MRSA", 1.2),
    "MDR-TB": _get_float("MDR_PATHOGEN_MDR_TB", 2.0),
    "VRE": _get_float("MDR_PATHOGEN_VRE", 1.5),
    "CRE": _get_float("MDR_PATHOGEN_CRE", 1.8),
    "ESBL": _get_float("MDR_PATHOGEN_ESBL", 1.3),
    "Other": _get_float("MDR_PATHOGEN_OTHER", 1.0),
}

# Vulnerability Factors (V) - Based on mask/PPE status
MDR_VULNERABILITY_NO_MASK = _get_float("MDR_VULNERABILITY_NO_MASK", 1.0)
MDR_VULNERABILITY_WITH_MASK = _get_float("MDR_VULNERABILITY_WITH_MASK", 0.3)

# Distance estimation from camera
MDR_PIXELS_PER_METER = _get_float("MDR_PIXELS_PER_METER", 150.0)
MDR_MIN_DISTANCE_METERS = _get_float("MDR_MIN_DISTANCE_METERS", 0.5)

# Risk thresholds
MDR_RISK_LOW_THRESHOLD = _get_float("MDR_RISK_LOW_THRESHOLD", 20.0)
MDR_RISK_MEDIUM_THRESHOLD = _get_float("MDR_RISK_MEDIUM_THRESHOLD", 40.0)
MDR_RISK_HIGH_THRESHOLD = _get_float("MDR_RISK_HIGH_THRESHOLD", 60.0)
MDR_RISK_CRITICAL_THRESHOLD = _get_float("MDR_RISK_CRITICAL_THRESHOLD", 80.0)

__all__ = [
    "register_settings",
    "recognition_settings",
    "dual_view_settings",
    "contact_settings",
    "collision_settings",
]
