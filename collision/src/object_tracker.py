from typing import Dict, List, Tuple
import numpy as np


def calculate_iou_boxes(box1: Tuple[float, float, float, float], 
                        box2: Tuple[float, float, float, float]) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


class SimpleTracker:
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 0
        self.tracks: Dict[int, Dict] = {}
    
    def update(self, detections: List[Tuple[np.ndarray, float, str]]) -> List[Tuple[np.ndarray, float, str]]:
        det_boxes = []
        det_confs = []
        for bbox, conf, _ in detections:
            if isinstance(bbox, np.ndarray):
                det_boxes.append(tuple(bbox))
            else:
                det_boxes.append(bbox)
            det_confs.append(conf)
        
        matched_tracks = set()
        matched_dets = set()
        matches = []
        
        for track_id, track_info in self.tracks.items():
            track_bbox = track_info['bbox']
            
            for det_idx, det_bbox in enumerate(det_boxes):
                iou = calculate_iou_boxes(track_bbox, det_bbox)
                
                if iou >= self.iou_threshold:
                    matches.append((track_id, det_idx, iou))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        
        assignments = {}
        for track_id, det_idx, iou in matches:
            if track_id not in matched_tracks and det_idx not in matched_dets:
                assignments[det_idx] = track_id
                matched_tracks.add(track_id)
                matched_dets.add(det_idx)
        
        for det_idx, track_id in assignments.items():
            self.tracks[track_id] = {
                'bbox': det_boxes[det_idx],
                'age': 0,
                'confidence': det_confs[det_idx],
            }
        
        new_track_ids = {}
        for det_idx in range(len(det_boxes)):
            if det_idx not in matched_dets:
                new_id = self.next_id
                self.next_id += 1
                
                self.tracks[new_id] = {
                    'bbox': det_boxes[det_idx],
                    'age': 0,
                    'confidence': det_confs[det_idx],
                }
                new_track_ids[det_idx] = new_id
        
        tracks_to_remove = []
        for track_id in self.tracks:
            if track_id not in matched_tracks:
                self.tracks[track_id]['age'] += 1
                
                if self.tracks[track_id]['age'] > self.max_age:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        tracked_detections = []
        for det_idx, (bbox, conf, _) in enumerate(detections):
            if det_idx in assignments:
                track_id = assignments[det_idx]
            elif det_idx in new_track_ids:
                track_id = new_track_ids[det_idx]
            else:
                track_id = self.next_id
                self.next_id += 1
            
            tracked_identity = f"Person_{track_id}"
            tracked_detections.append((bbox, conf, tracked_identity))
        
        return tracked_detections
    
    def reset(self):
        self.tracks.clear()
        self.next_id = 0
