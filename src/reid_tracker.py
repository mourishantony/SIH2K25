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
        nms_iou: float = 0.35,
        box_shrink: float = 0.1,
        device: Optional[str] = None,
    ) -> None:
        self.detector = YOLO(model_path)
        if device is not None:
            self.detector.to(device)
        self.det_conf = detection_confidence
        self.nms_iou = max(0.05, min(0.9, nms_iou))
        self.box_shrink = max(0.0, min(0.4, box_shrink))
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
        results = self.detector.predict(
            frame,
            conf=self.det_conf,
            iou=self.nms_iou,
            classes=[0],
            verbose=False,
        )
        detections: List[Tuple[BBox, float, str]] = []
        for result in results:
            if not hasattr(result, "boxes"):
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.det_conf:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append((self._tighten_bbox((x1, y1, x2, y2)), conf, "person"))
        return detections

    def _tighten_bbox(self, bbox: BBox) -> BBox:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0 or self.box_shrink <= 0:
            return bbox
        dx = int(width * self.box_shrink / 2)
        dy = int(height * self.box_shrink / 2)
        nx1 = x1 + dx
        ny1 = y1 + dy
        nx2 = x2 - dx
        ny2 = y2 - dy
        if nx2 - nx1 < 4:
            mid_x = (x1 + x2) // 2
            nx1 = mid_x - 2
            nx2 = mid_x + 2
        if ny2 - ny1 < 4:
            mid_y = (y1 + y2) // 2
            ny1 = mid_y - 2
            ny2 = mid_y + 2
        return (nx1, ny1, nx2, ny2)

    def update(self, frame: np.ndarray) -> Sequence[TrackInfo]:
        detections = self._detect(frame)
        tracks = self.tracker.update_tracks(detections, frame=frame)
        output: List[TrackInfo] = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            det_conf = float(getattr(track, "det_confidence", 0.0))
            tightened = self._tighten_bbox((x1, y1, x2, y2))
            output.append(TrackInfo(track_id=track.track_id, bbox=tightened, confidence=det_conf))
        return output


__all__ = ["PersonTracker", "TrackInfo"]
