from __future__ import annotations

from typing import List, Tuple
import numpy as np
from ultralytics import YOLO
import torch


class PersonDetectorYOLOv8:

    def __init__(self, model_size: str = "n"):
        print(f"🚀 Loading YOLOv8{model_size} person detector with GPU optimization...")
        
        self.model = YOLO(f"yolov8{model_size}.pt")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU ACCELERATION ENABLED")
            print(f"   GPU: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
            print(f"   CUDA Version: {torch.version.cuda}")
            
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        else:
            print("⚠️ GPU not available, using CPU (much slower)")
        
        self.model.to(self.device)
        
        self.person_class_id = 0

    def detect(self, frame: np.ndarray) -> List[Tuple[np.ndarray, float, str]]:
        results = self.model.predict(
            frame,
            conf=0.25,
            classes=[self.person_class_id],
            verbose=False,
            device=self.device,
            half=True if self.device == "cuda" else False,
            imgsz=640,
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                
                bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
                detections.append((bbox, confidence, "unknown"))
        
        return detections


_detector_instance = None


def get_person_detector_yolov8(model_size: str = "n") -> PersonDetectorYOLOv8:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PersonDetectorYOLOv8(model_size=model_size)
    return _detector_instance
