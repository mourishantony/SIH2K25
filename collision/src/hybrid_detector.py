from typing import List, Tuple
import numpy as np
from person_detector import PersonDetector
from vision import get_analyzer


class HybridDetector:
    
    def __init__(self, det_size=(640, 640), use_gpu=False, min_confidence=0.3):
        self.face_analyzer = get_analyzer(det_size, use_gpu)
        self.person_detector = PersonDetector(confidence_threshold=min_confidence)
        self.min_confidence = min_confidence
    
    def detect(self, frame: np.ndarray) -> List[Tuple[np.ndarray, float, str]]:
        results = []
        
        try:
            faces = self.face_analyzer.get(frame)
            for idx, face in enumerate(faces):
                if face.det_score >= self.min_confidence:
                    bbox = face.bbox.astype(np.float32)
                    results.append((bbox, float(face.det_score), f"Face_{idx}"))
        except Exception as e:
            print(f"Face detection error: {e}")
        
        try:
            people = self.person_detector.detect(frame)
            results.extend(people)
        except Exception as e:
            print(f"Person detection error: {e}")
        
        results = self._remove_overlapping_detections(results)
        
        return results
    
    def _remove_overlapping_detections(
        self, 
        detections: List[Tuple[np.ndarray, float, str]]
    ) -> List[Tuple[np.ndarray, float, str]]:
        if len(detections) <= 1:
            return detections
        
        detections = sorted(detections, key=lambda x: x[1], reverse=True)
        
        filtered = []
        for i, (bbox1, conf1, id1) in enumerate(detections):
            is_duplicate = False
            
            for bbox2, conf2, id2 in filtered:
                if self._is_inside(bbox1, bbox2) or self._is_inside(bbox2, bbox1):
                    if "Face" in id1 and "Person" in id2:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append((bbox1, conf1, id1))
        
        return filtered
    
    def _is_inside(self, bbox1: np.ndarray, bbox2: np.ndarray, threshold=0.7) -> bool:
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return False
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        
        return (inter_area / bbox1_area) > threshold


def get_hybrid_detector(det_size=(640, 640), use_gpu=False, min_confidence=0.3):
    return HybridDetector(det_size, use_gpu, min_confidence)
