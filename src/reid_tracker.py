"""Person detection + tracking powered by Ultralytics YOLO, DeepSORT, and OSNet."""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
BBox = Tuple[int, int, int, int]


def _ensure_torchreid_namespace() -> None:
    """Expose torchreid.reid.utils as torchreid.utils for DeepSORT compatibility."""
    try:
        import torchreid  # type: ignore  # noqa: F401
    except ImportError:
        return

    if "torchreid.utils" in sys.modules:
        return

    try:
        utils_module = importlib.import_module("torchreid.reid.utils")
    except ModuleNotFoundError:
        return

    sys.modules["torchreid.utils"] = utils_module



@dataclass
class TrackInfo:
    track_id: int
    bbox: BBox
    confidence: float


class PersonTracker:
    """Detect people via YOLOv8 and maintain tracks using DeepSORT + OSNet."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        detection_confidence: float = 0.35,
        max_iou_distance: float = 0.7,
        max_age: int = 30,
        n_init: int = 3,
        embedder_gpu: bool = False,
        embedder_model: str = "osnet_x0_25",
    ) -> None:
        self.detector = YOLO(model_path)
        self.det_conf = detection_confidence
        _ensure_torchreid_namespace()
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            embedder="torchreid",
            embedder_model_name=embedder_model,
            embedder_gpu=embedder_gpu,
            half=True,
        )

    def _detect(self, frame: np.ndarray) -> List[Tuple[BBox, float, str]]:
        results = self.detector.predict(frame, conf=self.det_conf, classes=[0], verbose=False)
        detections: List[Tuple[BBox, float, str]] = []
        for result in results:
            if not hasattr(result, "boxes"):
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.det_conf:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(((x1, y1, x2, y2), conf, "person"))
        return detections

    def update(self, frame: np.ndarray) -> Sequence[TrackInfo]:
        detections = self._detect(frame)
        tracks = self.tracker.update_tracks(detections, frame=frame)
        output: List[TrackInfo] = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            det_conf = float(getattr(track, "det_confidence", 0.0))
            output.append(TrackInfo(track_id=track.track_id, bbox=(x1, y1, x2, y2), confidence=det_conf))
        return output


__all__ = ["PersonTracker", "TrackInfo"]
