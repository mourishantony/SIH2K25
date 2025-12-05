"""
MDR Risk Score Calculator

Implements the infection risk formula:
    R = (T × P × V) / D²

Where:
- R: Infection Risk Score (0 to 100)
- T: Duration of contact in minutes
- P: Pathogen Factor (e.g., MRSA = 1.2, MDR-TB = 2.0)
- D: Distance between persons in meters
- V: Vulnerability multiplier (based on mask/PPE status)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from enum import Enum

from config import (
    MDR_PATHOGEN_FACTORS,
    MDR_VULNERABILITY_NO_MASK,
    MDR_VULNERABILITY_WITH_MASK,
    MDR_PIXELS_PER_METER,
    MDR_MIN_DISTANCE_METERS,
    MDR_RISK_LOW_THRESHOLD,
    MDR_RISK_MEDIUM_THRESHOLD,
    MDR_RISK_HIGH_THRESHOLD,
    MDR_RISK_CRITICAL_THRESHOLD,
)


class RiskLevel(Enum):
    """Risk level categories based on calculated score."""
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class MDRRiskResult:
    """Result of MDR risk calculation."""
    risk_score: float  # 0-100 scale
    risk_level: RiskLevel
    duration_minutes: float
    pathogen_factor: float
    pathogen_type: str
    vulnerability_factor: float
    distance_meters: float
    is_masked_mdr: bool
    is_masked_contact: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 2),
            "risk_level": self.risk_level.value,
            "duration_minutes": round(self.duration_minutes, 2),
            "pathogen_factor": self.pathogen_factor,
            "pathogen_type": self.pathogen_type,
            "vulnerability_factor": round(self.vulnerability_factor, 2),
            "distance_meters": round(self.distance_meters, 2),
            "is_masked_mdr": self.is_masked_mdr,
            "is_masked_contact": self.is_masked_contact,
        }


def get_pathogen_factor(pathogen_type: str) -> float:
    """Get the pathogen factor for a given pathogen type."""
    return MDR_PATHOGEN_FACTORS.get(pathogen_type, MDR_PATHOGEN_FACTORS.get("Other", 1.0))


def get_available_pathogens() -> Dict[str, float]:
    """Get all available pathogen types and their factors."""
    return MDR_PATHOGEN_FACTORS.copy()


def get_vulnerability_factor(
    is_contact_masked: bool,
    is_mdr_masked: bool = False,
) -> float:
    """
    Calculate vulnerability factor based on mask status.
    
    If the contact person wears a mask, V is reduced.
    If MDR patient also wears mask, further reduction.
    """
    base_v = MDR_VULNERABILITY_WITH_MASK if is_contact_masked else MDR_VULNERABILITY_NO_MASK
    
    # If MDR patient is also wearing mask, reduce risk further
    if is_mdr_masked:
        base_v *= 0.7  # 30% additional reduction
    
    return base_v


def estimate_distance_from_pixels(
    pixel_distance: float,
    box1_height: Optional[float] = None,
    box2_height: Optional[float] = None,
) -> float:
    """
    Estimate real-world distance from pixel distance.
    
    Uses a simple conversion based on average person height and box size.
    If bounding box heights are provided, uses them for better estimation.
    """
    # If we have bounding box heights, use them to estimate distance
    if box1_height and box2_height:
        # Average person height ~1.7m, use box height to estimate scale
        avg_box_height = (box1_height + box2_height) / 2
        # Assuming average person is ~1.7m, estimate pixels per meter
        if avg_box_height > 0:
            estimated_ppm = avg_box_height / 1.7
            distance_meters = pixel_distance / estimated_ppm if estimated_ppm > 0 else pixel_distance / MDR_PIXELS_PER_METER
        else:
            distance_meters = pixel_distance / MDR_PIXELS_PER_METER
    else:
        # Use default conversion
        distance_meters = pixel_distance / MDR_PIXELS_PER_METER
    
    # Ensure minimum distance to avoid division issues
    return max(distance_meters, MDR_MIN_DISTANCE_METERS)


def calculate_mdr_risk(
    duration_seconds: float,
    pixel_distance: float,
    pathogen_type: str,
    is_contact_masked: bool = False,
    is_mdr_masked: bool = False,
    box1_height: Optional[float] = None,
    box2_height: Optional[float] = None,
) -> MDRRiskResult:
    """
    Calculate MDR infection risk score using the formula:
    R = (T × P × V) / D²
    
    Args:
        duration_seconds: Contact duration in seconds
        pixel_distance: Distance between persons in pixels
        pathogen_type: Type of MDR pathogen (MRSA, MDR-TB, etc.)
        is_contact_masked: Whether the contact person is wearing a mask
        is_mdr_masked: Whether the MDR patient is wearing a mask
        box1_height: Height of first person's bounding box (for distance estimation)
        box2_height: Height of second person's bounding box (for distance estimation)
    
    Returns:
        MDRRiskResult with calculated risk score and details
    """
    # Convert duration to minutes
    T = duration_seconds / 60.0
    
    # Get pathogen factor
    P = get_pathogen_factor(pathogen_type)
    
    # Get vulnerability factor based on mask status
    V = get_vulnerability_factor(is_contact_masked, is_mdr_masked)
    
    # Estimate distance in meters
    D = estimate_distance_from_pixels(pixel_distance, box1_height, box2_height)
    
    # Calculate risk score: R = (T × P × V) / D²
    # Multiply by scaling factor to get 0-100 range
    scaling_factor = 10.0  # Adjust this to tune the output range
    R = (T * P * V * scaling_factor) / (D * D)
    
    # Clamp to 0-100 range
    risk_score = max(0.0, min(100.0, R))
    
    # Determine risk level
    risk_level = get_risk_level(risk_score)
    
    return MDRRiskResult(
        risk_score=risk_score,
        risk_level=risk_level,
        duration_minutes=T,
        pathogen_factor=P,
        pathogen_type=pathogen_type,
        vulnerability_factor=V,
        distance_meters=D,
        is_masked_mdr=is_mdr_masked,
        is_masked_contact=is_contact_masked,
    )


def get_risk_level(score: float) -> RiskLevel:
    """Get risk level category from score."""
    if score < MDR_RISK_LOW_THRESHOLD:
        return RiskLevel.SAFE
    elif score < MDR_RISK_MEDIUM_THRESHOLD:
        return RiskLevel.LOW
    elif score < MDR_RISK_HIGH_THRESHOLD:
        return RiskLevel.MEDIUM
    elif score < MDR_RISK_CRITICAL_THRESHOLD:
        return RiskLevel.HIGH
    else:
        return RiskLevel.CRITICAL


def get_risk_color(level: RiskLevel) -> Tuple[int, int, int]:
    """Get BGR color for risk level (for OpenCV drawing)."""
    colors = {
        RiskLevel.SAFE: (0, 200, 0),      # Green
        RiskLevel.LOW: (0, 255, 255),     # Yellow
        RiskLevel.MEDIUM: (0, 165, 255),  # Orange
        RiskLevel.HIGH: (0, 0, 255),      # Red
        RiskLevel.CRITICAL: (128, 0, 128), # Purple
    }
    return colors.get(level, (255, 255, 255))


def format_risk_display(result: MDRRiskResult) -> str:
    """Format risk result for display on video feed."""
    return (
        f"MDR Risk: {result.risk_score:.1f}% [{result.risk_level.value}] "
        f"| T={result.duration_minutes:.1f}min "
        f"| P={result.pathogen_type}({result.pathogen_factor}) "
        f"| D={result.distance_meters:.1f}m "
        f"| V={result.vulnerability_factor:.2f}"
    )


__all__ = [
    "RiskLevel",
    "MDRRiskResult",
    "get_pathogen_factor",
    "get_available_pathogens",
    "get_vulnerability_factor",
    "estimate_distance_from_pixels",
    "calculate_mdr_risk",
    "get_risk_level",
    "get_risk_color",
    "format_risk_display",
]
