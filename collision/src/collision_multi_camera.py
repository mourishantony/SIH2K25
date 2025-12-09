
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from rich import print as rprint

from collision_detector import BoundingBox
from object_tracker import SimpleTracker


try:
    from person_detector_yolov8 import get_person_detector_yolov8
    USE_YOLOV8 = True
    rprint("✅ [green]YOLOv8 available for GPU acceleration[/green]")
except ImportError:
    USE_YOLOV8 = False
    rprint("⚠️ [yellow]YOLOv8 not available, will use YOLOv3[/yellow]")

from person_detector import get_person_detector


class CollisionCameraStream:

    def __init__(
        self,
        source: int | str,
        name: str,
        detector,
        min_confidence: float,
    ):
        self.source = source
        self.name = name
        self.detector = detector
        self.min_confidence = min_confidence
        self.tracker = SimpleTracker()
        
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open {self.name} source: {source}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        rprint(f"[green]{self.name} initialized: {self.width}x{self.height} @ {self.fps:.1f}fps[/green]")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        return ret, frame if ret else None

    def detect_people(self, frame: np.ndarray) -> List[BoundingBox]:
        detections = self.detector.detect(frame)
        
        tracked_detections = self.tracker.update(detections)
        
        bboxes = []
        for detection in tracked_detections:
            bbox_coords, confidence, person_id = detection
            x1, y1, x2, y2 = bbox_coords.astype(int)
            
            bbox = BoundingBox(
                x1=x1,
                y1=y1, 
                x2=x2,
                y2=y2,
                person_name=person_id,
                confidence=confidence,
            )
            bboxes.append(bbox)
        
        return bboxes

    def get_frame_dimensions(self) -> Tuple[int, int]:
        return self.width, self.height

    def release(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


class CollisionDualCameraManager:

    def __init__(
        self,
        source1: int | str,
        source2: int | str,
        det_size: Tuple[int, int] = (640, 640),
        use_gpu: bool = True,
        min_confidence: float = 0.3,
        threshold: float = 0.5,
    ):
        if USE_YOLOV8 and use_gpu:
            rprint("[🚀 green]USING YOLOv8 with RTX 3050 GPU ACCELERATION[🚀 /green]")
            detector = get_person_detector_yolov8(model_size="s")  
        elif USE_YOLOV8:
            rprint("[cyan]Using YOLOv8 CPU mode (GPU disabled)[/]")
            detector = get_person_detector_yolov8(model_size="n")
        else:
            rprint("[yellow]Fallback: Using YOLOv3 person detection[/]")
            detector = get_person_detector(confidence_threshold=min_confidence)
        
        self.stream1 = CollisionCameraStream(source1, "Camera 1", detector, min_confidence)
        self.stream2 = CollisionCameraStream(source2, "Camera 2", detector, min_confidence)
        
        self.frame_count = 0
        
        rprint(f"[bold green]Collision Detection System Ready[/bold green]")
        rprint(f"[cyan]Stream 1: {source1}[/cyan]")
        rprint(f"[cyan]Stream 2: {source2}[/cyan]")

    def read_synchronized_frames(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        ret1, frame1 = self.stream1.read_frame()
        ret2, frame2 = self.stream2.read_frame()
        
        success = ret1 and ret2
        if success:
            self.frame_count += 1
            
        return success, frame1, frame2

    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[List[BoundingBox], List[BoundingBox]]:
        bboxes1 = self.stream1.detect_people(frame1)
        bboxes2 = self.stream2.detect_people(frame2)
        
        return bboxes1, bboxes2

    def get_frame_dimensions(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        dims1 = self.stream1.get_frame_dimensions()
        dims2 = self.stream2.get_frame_dimensions()
        return dims1, dims2

    def release(self):
        self.stream1.release()
        self.stream2.release()
        rprint("[yellow]Camera resources released[/yellow]")


DualCameraManager = CollisionDualCameraManager