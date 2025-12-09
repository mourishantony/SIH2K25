import json
import math
from typing import Tuple


def center_of_bbox(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return cx, cy


def pixel_distance(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    (x1, y1) = center_of_bbox(bbox1)
    (x2, y2) = center_of_bbox(bbox2)
    return math.hypot(x2 - x1, y2 - y1)


def load_fx_from_calibration(calib_json_path: str = "data/calibration_cam.json") -> float:
    with open(calib_json_path, "r") as f:
        calib = json.load(f)
    if "fx" in calib:
        return float(calib["fx"])
    K = calib.get("camera_matrix")
    if not K:
        raise ValueError("camera_matrix not found in calibration JSON")
    
    return float(K[0][0])


def estimate_depth_from_bbox(bbox: Tuple[int, int, int, int], f_px: float, person_height_m: float = 1.70) -> float:
    h_px = max(1.0, (bbox[3] - bbox[1]))
    return (f_px * person_height_m) / h_px


def real_world_distance_meters(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int],
    f_px: float,
    person_height_m: float = 1.70,
) -> float:
    Z1 = estimate_depth_from_bbox(bbox1, f_px, person_height_m)
    Z2 = estimate_depth_from_bbox(bbox2, f_px, person_height_m)
    Zavg = (Z1 + Z2) / 2.0
    scale_m_per_px = Zavg / f_px
    (x1, y1) = center_of_bbox(bbox1)
    (x2, y2) = center_of_bbox(bbox2)
    dx_m = (x2 - x1) * scale_m_per_px
    dy_m = (y2 - y1) * scale_m_per_px
    return math.hypot(dx_m, dy_m)
