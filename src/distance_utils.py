"""Distance calculation utilities using camera calibration.

This module provides functions to calculate real-world distances between
detected persons using the pinhole camera model and camera calibration data.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional, Tuple

# Default calibration file paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CALIBRATION_PATHS = [
    ROOT_DIR / "collision" / "data" / "calibration_cam.json",
    ROOT_DIR / "data" / "calibration_cam.json",
    ROOT_DIR / "calibration_cam.json",
]

# Cache for calibration data
_calibration_cache: Optional[dict] = None
_fx_cache: Optional[float] = None


def center_of_bbox(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Get the center point of a bounding box.
    
    Args:
        bbox: Tuple of (x1, y1, x2, y2) coordinates
        
    Returns:
        Tuple of (center_x, center_y)
    """
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return cx, cy


def pixel_distance(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate pixel distance between centers of two bounding boxes.
    
    Args:
        bbox1: First bounding box (x1, y1, x2, y2)
        bbox2: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        Distance in pixels
    """
    (x1, y1) = center_of_bbox(bbox1)
    (x2, y2) = center_of_bbox(bbox2)
    return math.hypot(x2 - x1, y2 - y1)


def find_calibration_file() -> Optional[str]:
    """Find the calibration JSON file in known locations.
    
    Returns:
        Path to calibration file or None if not found
    """
    for path in DEFAULT_CALIBRATION_PATHS:
        if path.exists():
            return str(path)
    
    # Also check environment variable
    env_path = os.getenv("CALIBRATION_FILE_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    
    return None


def load_calibration(calib_json_path: Optional[str] = None) -> Optional[dict]:
    """Load camera calibration data from JSON file.
    
    Args:
        calib_json_path: Path to calibration JSON file (optional, will auto-detect)
        
    Returns:
        Calibration dictionary or None if not found
    """
    global _calibration_cache
    
    if _calibration_cache is not None:
        return _calibration_cache
    
    if calib_json_path is None:
        calib_json_path = find_calibration_file()
    
    if calib_json_path is None or not Path(calib_json_path).exists():
        return None
    
    try:
        with open(calib_json_path, "r") as f:
            _calibration_cache = json.load(f)
        return _calibration_cache
    except Exception:
        return None


def load_fx_from_calibration(calib_json_path: Optional[str] = None) -> Optional[float]:
    """Load the focal length (fx) from calibration file.
    
    Args:
        calib_json_path: Path to calibration JSON file (optional)
        
    Returns:
        Focal length in pixels or None if not found
    """
    global _fx_cache
    
    if _fx_cache is not None:
        return _fx_cache
    
    calib = load_calibration(calib_json_path)
    if calib is None:
        return None
    
    # Try direct fx field first
    if "fx" in calib:
        _fx_cache = float(calib["fx"])
        return _fx_cache
    
    # Try camera_matrix
    K = calib.get("camera_matrix")
    if K and len(K) >= 1 and len(K[0]) >= 1:
        _fx_cache = float(K[0][0])
        return _fx_cache
    
    return None


def estimate_depth_from_bbox(
    bbox: Tuple[int, int, int, int],
    f_px: float,
    person_height_m: float = 1.70
) -> float:
    """Estimate depth (Z) of a person using pinhole camera model.
    
    Uses the formula: Z ≈ (f_px * H_real) / H_pixels
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        f_px: Focal length in pixels
        person_height_m: Assumed real-world person height in meters
        
    Returns:
        Estimated depth in meters
    """
    h_px = max(1.0, (bbox[3] - bbox[1]))
    return (f_px * person_height_m) / h_px


def real_world_distance_meters(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int],
    f_px: Optional[float] = None,
    person_height_m: float = 1.70,
) -> Optional[float]:
    """Calculate approximate real-world distance between two persons in meters.
    
    Uses the pinhole camera model and assumed person height to estimate
    depth and calculate 3D distance between two detected persons.
    
    Args:
        bbox1: First person's bounding box (x1, y1, x2, y2)
        bbox2: Second person's bounding box (x1, y1, x2, y2)
        f_px: Focal length in pixels (optional, will auto-load from calibration)
        person_height_m: Assumed real-world person height in meters
        
    Returns:
        Distance in meters or None if calibration not available
    """
    if f_px is None:
        f_px = load_fx_from_calibration()
    
    if f_px is None:
        return None
    
    Z1 = estimate_depth_from_bbox(bbox1, f_px, person_height_m)
    Z2 = estimate_depth_from_bbox(bbox2, f_px, person_height_m)
    Zavg = (Z1 + Z2) / 2.0
    
    # Pixel-to-meter scale at average depth
    scale_m_per_px = Zavg / f_px
    
    (x1, y1) = center_of_bbox(bbox1)
    (x2, y2) = center_of_bbox(bbox2)
    
    dx_m = (x2 - x1) * scale_m_per_px
    dy_m = (y2 - y1) * scale_m_per_px
    
    return math.hypot(dx_m, dy_m)


def is_calibration_available() -> bool:
    """Check if camera calibration is available.
    
    Returns:
        True if calibration file exists and is valid
    """
    return load_fx_from_calibration() is not None


def get_calibration_info() -> dict:
    """Get information about the current calibration.
    
    Returns:
        Dictionary with calibration info or status
    """
    calib_path = find_calibration_file()
    fx = load_fx_from_calibration()
    
    return {
        "available": fx is not None,
        "calibration_file": calib_path,
        "fx": fx,
        "supported_paths": [str(p) for p in DEFAULT_CALIBRATION_PATHS],
    }


__all__ = [
    "center_of_bbox",
    "pixel_distance",
    "load_fx_from_calibration",
    "load_calibration",
    "estimate_depth_from_bbox",
    "real_world_distance_meters",
    "is_calibration_available",
    "get_calibration_info",
]
