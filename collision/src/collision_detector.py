from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import time

import numpy as np


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    person_name: str
    confidence: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class Collision:
    person1: str
    person2: str
    bbox1: BoundingBox
    bbox2: BoundingBox
    iou: float
    distance: float
    risk_level: str
    risk_score: float
    start_time: float = 0.0
    duration: float = 0.0
    frame_count: int = 0

    def __str__(self) -> str:
        return f"{self.person1} <-> {self.person2}: {self.risk_level} (score={self.risk_score:.2f}, duration={self.duration:.1f}s)"
    
    def get_collision_id(self) -> str:
        names = sorted([self.person1, self.person2])
        return f"{names[0]}_{names[1]}"


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = box1.area
    area2 = box2.area
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def calculate_distance(box1: BoundingBox, box2: BoundingBox) -> float:
    c1 = box1.center
    c2 = box2.center
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def calculate_risk_score(iou: float, distance: float, frame_diagonal: float) -> float:
    normalized_distance = distance / frame_diagonal if frame_diagonal > 0 else 1.0
    
    iou_weight = 0.7
    distance_weight = 0.3
    
    iou_risk = iou
    distance_risk = max(0, 1.0 - normalized_distance)
    
    risk_score = (iou_weight * iou_risk) + (distance_weight * distance_risk)
    return min(1.0, max(0.0, risk_score))


def get_risk_level(risk_score: float) -> str:
    if risk_score >= 0.7:
        return "CRITICAL"
    elif risk_score >= 0.5:
        return "HIGH"
    elif risk_score >= 0.3:
        return "MEDIUM"
    elif risk_score >= 0.1:
        return "LOW"
    else:
        return "SAFE"


def detect_collisions(
    bboxes: List[BoundingBox],
    iou_threshold: float = 0.1,
    distance_threshold: float = 200,
    frame_width: int = 640,
    frame_height: int = 480,
) -> List[Collision]:
    collisions: List[Collision] = []
    frame_diagonal = np.sqrt(frame_width ** 2 + frame_height ** 2)

    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            box1 = bboxes[i]
            box2 = bboxes[j]

            iou = calculate_iou(box1, box2)
            distance = calculate_distance(box1, box2)

            if iou >= iou_threshold or distance <= distance_threshold:
                risk_score = calculate_risk_score(iou, distance, frame_diagonal)
                risk_level = get_risk_level(risk_score)

                collision = Collision(
                    person1=box1.person_name,
                    person2=box2.person_name,
                    bbox1=box1,
                    bbox2=box2,
                    iou=iou,
                    distance=distance,
                    risk_level=risk_level,
                    risk_score=risk_score,
                    start_time=0.0,
                    duration=0.0,
                    frame_count=0,
                )
                collisions.append(collision)

    collisions.sort(key=lambda c: c.risk_score, reverse=True)
    return collisions


class CollisionTracker:
    
    def __init__(self, time_threshold: float = 10.0):
        self.time_threshold = time_threshold
        self.active_collisions = {}
        self.start_time = time.time()
    
    def update_collisions(self, current_collisions: List[Collision]) -> List[Collision]:
        current_time = time.time() - self.start_time
        current_ids = set()
        updated_collisions = []
        
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
            updated_collisions.append(collision)
        
        ended_collisions = set(self.active_collisions.keys()) - current_ids
        for collision_id in ended_collisions:
            del self.active_collisions[collision_id]
        
        return updated_collisions
    
    def get_significant_collisions(self, collisions: List[Collision]) -> List[Collision]:
        return [c for c in collisions if c.duration >= self.time_threshold]


def verify_collision_across_cameras(
    collisions_cam1: List[Collision],
    collisions_cam2: List[Collision],
    name_match_threshold: float = 0.8,
) -> List[Tuple[Collision, Collision]]:
    verified: List[Tuple[Collision, Collision]] = []

    for c1 in collisions_cam1:
        people1 = {c1.person1, c1.person2}
        
        for c2 in collisions_cam2:
            people2 = {c2.person1, c2.person2}
            
            overlap = len(people1 & people2)
            match_ratio = overlap / len(people1 | people2) if people1 | people2 else 0
            
            if match_ratio >= name_match_threshold:
                verified.append((c1, c2))

    return verified
