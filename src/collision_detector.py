from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class BoundingBox:
    """Normalized representation of a tracked person bounding box."""

    x1: int
    y1: int
    x2: int
    y2: int
    person_name: str
    confidence: float = 1.0

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass
class Collision:
    person1: str
    person2: str
    box1: BoundingBox
    box2: BoundingBox
    iou: float
    distance: float
    risk_level: str
    risk_score: float
    start_time: float = 0.0
    duration: float = 0.0
    frame_count: int = 0

    def get_collision_id(self) -> Tuple[str, str]:
        return tuple(sorted((self.person1, self.person2)))


def calculate_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    ax1, ay1, ax2, ay2 = box_a.x1, box_a.y1, box_a.x2, box_a.y2
    bx1, by1, bx2, by2 = box_b.x1, box_b.y1, box_b.x2, box_b.y2
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    if inter_w == 0 or inter_h == 0:
        return 0.0
    inter_area = inter_w * inter_h
    area_a = max(1, box_a.width * box_a.height)
    area_b = max(1, box_b.width * box_b.height)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def calculate_distance(box_a: BoundingBox, box_b: BoundingBox) -> float:
    ax, ay = box_a.center
    bx, by = box_b.center
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def calculate_risk_score(iou: float, distance: float, frame_diagonal: float) -> float:
    """Calculate risk score based on IoU and distance.
    
    When boxes overlap (IoU > 0), that's a strong indicator of close contact.
    When boxes are close but don't overlap, use distance-based risk.
    """
    normalized_distance = distance / frame_diagonal if frame_diagonal > 0 else 1.0
    
    # If there's any overlap, weight it heavily
    if iou > 0:
        iou_weight = 0.6
        distance_weight = 0.4
        iou_risk = max(0.0, min(1.0, iou * 2))  # Amplify IoU contribution
    else:
        # No overlap - rely more on distance
        iou_weight = 0.0
        distance_weight = 1.0
        iou_risk = 0.0
    
    # Distance risk: closer = higher risk (inverse relationship)
    # Use exponential decay for more sensitivity to close distances
    distance_risk = max(0.0, 1.0 - (normalized_distance ** 0.5))  # Square root for gentler decay
    
    risk_score = (iou_weight * iou_risk) + (distance_weight * distance_risk)
    return max(0.0, min(1.0, risk_score))


def get_risk_level(score: float) -> str:
    if score < 0.2:
        return "SAFE"
    if score < 0.4:
        return "LOW"
    if score < 0.6:
        return "MEDIUM"
    if score < 0.8:
        return "HIGH"
    return "CRITICAL"


def detect_collisions(
    bboxes: Sequence[BoundingBox],
    *,
    iou_threshold: float = 0.01,  # Very low - any overlap counts
    distance_threshold: float = 350.0,  # Increased - detect close proximity even without overlap
    frame_width: int = 640,
    frame_height: int = 480,
) -> List[Collision]:
    """Detect collisions (close contacts) between bounding boxes.
    
    A collision is detected if EITHER:
    - IoU >= iou_threshold (boxes overlap), OR
    - Distance between centers <= distance_threshold (boxes are close)
    """
    collisions: List[Collision] = []
    if not bboxes:
        return collisions
    frame_diagonal = math.sqrt(frame_width ** 2 + frame_height ** 2)
    for idx in range(len(bboxes)):
        for jdx in range(idx + 1, len(bboxes)):
            box1 = bboxes[idx]
            box2 = bboxes[jdx]
            iou = calculate_iou(box1, box2)
            distance = calculate_distance(box1, box2)
            
            # Detect collision if boxes overlap OR are close enough
            if iou < iou_threshold and distance > distance_threshold:
                continue
                
            risk_score = calculate_risk_score(iou, distance, frame_diagonal)
            risk_level = get_risk_level(risk_score)
            collisions.append(
                Collision(
                    person1=box1.person_name,
                    person2=box2.person_name,
                    box1=box1,
                    box2=box2,
                    iou=iou,
                    distance=distance,
                    risk_level=risk_level,
                    risk_score=risk_score,
                )
            )
    collisions.sort(key=lambda c: c.risk_score, reverse=True)
    return collisions


class CollisionTracker:
    """Track collision durations across frames for consistent alerting."""

    def __init__(self, time_threshold: float = 10.0) -> None:
        self.time_threshold = time_threshold
        self.active_collisions: Dict[Tuple[str, str], Collision] = {}
        self.start_reference = time.time()

    def update_collisions(self, current_collisions: Iterable[Collision]) -> List[Collision]:
        current_time = time.time() - self.start_reference
        current_ids = set()
        updated: List[Collision] = []
        for collision in current_collisions:
            collision_id = collision.get_collision_id()
            current_ids.add(collision_id)
            if collision_id in self.active_collisions:
                existing = self.active_collisions[collision_id]
                collision.start_time = existing.start_time
                collision.duration = current_time - existing.start_time
                collision.frame_count = existing.frame_count + 1
            else:
                collision.start_time = current_time
                collision.duration = 0.0
                collision.frame_count = 1
            self.active_collisions[collision_id] = collision
            updated.append(collision)
        for collision_id in list(self.active_collisions.keys()):
            if collision_id not in current_ids:
                del self.active_collisions[collision_id]
        return updated


def verify_collision_across_cameras(
    collisions_a: Sequence[Collision],
    collisions_b: Sequence[Collision],
) -> List[Tuple[Collision, Optional[Collision]]]:
    index_b: Dict[Tuple[str, str], Collision] = {collision.get_collision_id(): collision for collision in collisions_b}
    verified: List[Tuple[Collision, Optional[Collision]]] = []
    processed = set()
    for collision in collisions_a:
        collision_id = collision.get_collision_id()
        partner = index_b.get(collision_id)
        verified.append((collision, partner))
        processed.add(collision_id)
    for collision_id, collision in index_b.items():
        if collision_id in processed:
            continue
        verified.append((collision, None))
    return verified


__all__ = [
    "BoundingBox",
    "Collision",
    "CollisionTracker",
    "calculate_iou",
    "calculate_distance",
    "calculate_risk_score",
    "get_risk_level",
    "detect_collisions",
    "verify_collision_across_cameras",
]
