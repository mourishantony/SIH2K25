from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from rich import print as rprint

from collision_detector import BoundingBox
from face_db import flatten_registry, load_facebank
from vision import get_analyzer
from person_detector import get_person_detector
from hybrid_detector import get_hybrid_detector
from object_tracker import SimpleTracker

try:
    from person_detector_yolov8 import get_person_detector_yolov8
    USE_YOLOV8 = True
except ImportError:
    USE_YOLOV8 = False


class CameraStream:

    def __init__(
        self,
        source: int | str,
        name: str,
        detector,
        min_confidence: float,
        threshold: float,
    ):
        self.source = source
        self.name = name
        self.detector = detector
        self.min_confidence = min_confidence
        self.threshold = threshold
        
        self.tracker = SimpleTracker(iou_threshold=0.3, max_age=30)
        
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source)
        else:
            self.cap = cv2.VideoCapture(str(source))
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open {name}: {source}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        self.frame_skip = 2
        self.frame_count = 0
        self.last_detections = []
        
        rprint(f"[green]Initialized {name}: {self.width}x{self.height} @ {self.fps:.1f}fps[/]")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        return ret, frame if ret else None

    def detect_and_recognize(
        self,
        frame: np.ndarray,
        names: np.ndarray,
        embeddings: np.ndarray,
    ) -> List[BoundingBox]:
        self.frame_count += 1
        
        if self.frame_count % self.frame_skip != 0:
            return self.last_detections
        
        detections = self.detector.detect(frame)
        
        tracked_detections = self.tracker.update(detections)
        
        bboxes: List[BoundingBox] = []

        for bbox_coords, confidence, identity in tracked_detections:
            if confidence < self.min_confidence:
                continue

            x1, y1, x2, y2 = map(int, bbox_coords)

            bbox = BoundingBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                person_name=identity,
                confidence=confidence,
            )
            bboxes.append(bbox)

        self.last_detections = bboxes
        return bboxes

    def _predict_identity(
        self,
        embedding: np.ndarray,
        names: np.ndarray,
        embeddings: np.ndarray,
    ) -> Tuple[str, float]:
        if embeddings.size == 0:
            return "Unknown", 0.0
        
        similarities = embeddings @ embedding
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        
        if best_score < self.threshold:
            return "Unknown", best_score
        
        return str(names[best_idx]), best_score

    def release(self):
        self.cap.release()


class DualCameraManager:

    def __init__(
        self,
        source1: int | str,
        source2: int | str,
        det_size: int,
        use_gpu: bool,
        min_confidence: float,
        threshold: float,
    ):
        if USE_YOLOV8 and use_gpu:
            rprint("[🚀 green]USING YOLOv8 with RTX 3050 GPU ACCELERATION[🚀 /green]")
            detector = get_person_detector_yolov8(model_size="s")
        elif USE_YOLOV8:
            rprint("[cyan]Using YOLOv8 CPU mode (GPU disabled)[/]")
            detector = get_person_detector_yolov8(model_size="n")
        else:
            rprint("[yellow]Fallback: Using hybrid detection (face + person)[/]")
            detector = get_hybrid_detector(
                det_size=(det_size, det_size),
                use_gpu=use_gpu,
                min_confidence=min_confidence
            )
        
        self.names = np.array([])
        self.embeddings = np.empty((0, 512), dtype=np.float32)

        self.stream1 = CameraStream(
            source=source1,
            name="Camera 1",
            detector=detector,
            min_confidence=min_confidence,
            threshold=threshold,
        )
        self.stream2 = CameraStream(
            source=source2,
            name="Camera 2",
            detector=detector,
            min_confidence=min_confidence,
            threshold=threshold,
        )

        self.frame_count = 0

    def read_synchronized_frames(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        ret1, frame1 = self.stream1.read_frame()
        ret2, frame2 = self.stream2.read_frame()

        if not ret1 or not ret2:
            return False, None, None

        self.frame_count += 1
        return True, frame1, frame2

    def process_frames(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> Tuple[List[BoundingBox], List[BoundingBox]]:
        bboxes1 = self.stream1.detect_and_recognize(frame1, self.names, self.embeddings)
        bboxes2 = self.stream2.detect_and_recognize(frame2, self.names, self.embeddings)
        
        return bboxes1, bboxes2

    def get_frame_dimensions(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (
            (self.stream1.width, self.stream1.height),
            (self.stream2.width, self.stream2.height),
        )

    def release(self):
        self.stream1.release()
        self.stream2.release()
