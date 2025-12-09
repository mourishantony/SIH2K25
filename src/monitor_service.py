"""Contact Monitor Service - Web-integrated version of monitor_contacts.py."""
from __future__ import annotations

import asyncio
import base64
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, AsyncGenerator, Any

import cv2
import numpy as np

# Import MongoDB-based modules
from alert_system_mongo import AlertSystemMongo as AlertSystem
from collision_detector import (
    BoundingBox,
    Collision,
    CollisionTracker,
    detect_collisions,
    verify_collision_across_cameras,
)
from config import collision_settings, contact_settings, dual_view_settings, recognition_settings
from contact_store_mongo import ContactLedgerMongo as ContactLedger
from email_alerter_mongo import EmailAlerterMongo as EmailAlerter, MDRContactAlert
from face_db_mongo import flatten_registry, load_facebank
from mask_classifier import MaskClassifier
from mdr_tracker_mongo import is_mdr_patient, get_mdr_patients as load_mdr_patients
from person_risk_store import get_bidirectional_risks, update_bidirectional_risks
from reid_tracker import PersonTracker, TrackInfo
from vision import get_analyzer
from unknown_tracker_mongo import get_unknown_tracker, UnknownPersonTracker

BBox = Tuple[int, int, int, int]


def get_local_timestamp_iso() -> str:
    """Get current local timestamp in ISO format with timezone offset."""
    local_now = datetime.now().astimezone()
    return local_now.isoformat()


@dataclass
class PairState:
    cumulative: float = 0.0
    active: bool = False
    start_iso: Optional[str] = None
    end_iso: Optional[str] = None
    person_a: Optional[str] = None
    person_b: Optional[str] = None
    cumulative_risk_a: float = 0.0
    cumulative_risk_b: float = 0.0
    involves_mdr: bool = False
    mdr_patient: Optional[str] = None
    other_person: Optional[str] = None
    mdr_alert_sent: bool = False
    start_timestamp: float = 0.0
    pathogen_type: str = "Other"
    pathogen_factor: float = 1.0
    mdr_risk_score: float = 0.0
    last_pixel_distance: float = 0.0
    last_distance_meters: Optional[float] = None  # Real-world distance in meters
    min_distance_meters: Optional[float] = None  # Minimum distance recorded during contact
    mdr_patient_masked: bool = False
    other_person_masked: bool = False
    involves_unknown: bool = False
    unknown_temp_id: Optional[str] = None
    unknown_track_id: Optional[int] = None
    unknown_alert_sent: bool = False


@dataclass
class MaskState:
    probability: float
    timestamp: float


class MaskMemory:
    def __init__(self, decay_seconds: float) -> None:
        self.decay = decay_seconds
        self._values: Dict[str, MaskState] = {}

    def update(self, name: str, probability: float, timestamp: float) -> None:
        self._values[name] = MaskState(probability=probability, timestamp=timestamp)

    def probability(self, name: str, timestamp: float) -> float:
        state = self._values.get(name)
        if state is None:
            return 0.0
        if timestamp - state.timestamp > self.decay:
            return 0.0
        return state.probability


class ViewPipeline:
    """Process frames from a single camera view."""
    
    def __init__(
        self,
        *,
        label: str,
        tracker: PersonTracker,
        capture: cv2.VideoCapture,
    ) -> None:
        self.label = label
        self.tracker = tracker
        self.capture = capture
        self.track_identities: Dict[int, str] = {}
        self.identity_claims: Dict[str, int] = {}
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps > 1.0 else 30.0
        self.frame_interval = 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = self.capture.read()
        if not ok:
            return False, None
        return True, frame

    def process(
        self,
        frame: np.ndarray,
        *,
        analyzer,
        names: np.ndarray,
        embeddings: np.ndarray,
        min_confidence: float,
        threshold: float,
        mask_classifier: MaskClassifier,
        mask_memory: MaskMemory,
        timestamp: float,
        unknown_tracker: Optional[UnknownPersonTracker] = None,
    ) -> Tuple[np.ndarray, Dict[str, BBox]]:
        """Process frame and return annotated frame with named bounding boxes."""
        tracks = self.tracker.update(frame)
        named_boxes: Dict[str, BBox] = {}
        faces = analyzer.get(frame)
        
        unknown_track_faces: Dict[int, Tuple[np.ndarray, float, np.ndarray]] = {}
        recognized_faces: Dict[str, BBox] = {}
        
        for face in faces:
            if face.det_score < min_confidence:
                continue
            x1, y1, x2, y2 = map(int, face.bbox)
            embedding = face.normed_embedding.astype(np.float32)
            identity, score = self._predict_identity(embedding, names, embeddings, threshold)
            color = (0, 200, 100) if identity != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Get face landmarks for quality validation (InsightFace provides 5-point landmarks)
            face_landmarks = getattr(face, 'kps', None)  # kps = keypoints (5-point landmarks)
            if face_landmarks is None:
                face_landmarks = getattr(face, 'landmark', None)  # Alternative attribute name
            face_bbox = (x1, y1, x2, y2)
            
            if identity == "Unknown":
                matched = self._match_face_to_track((x1, y1, x2, y2), tracks)
                if matched is not None and unknown_tracker is not None:
                    face_crop = self._crop(frame, (x1, y1, x2, y2))
                    if face_crop.size > 0:
                        unknown_track_faces[matched.track_id] = (face_crop, face.det_score, embedding)
                    
                    state = unknown_tracker.register_unknown(
                        track_id=matched.track_id,
                        face_crop=face_crop if face_crop.size > 0 else None,
                        face_score=face.det_score,
                        face_embedding=embedding,
                        monotonic_timestamp=timestamp,
                        face_landmarks=face_landmarks,
                        face_bbox=face_bbox,
                    )
                    
                    if state is not None:
                        temp_id = state.temp_id
                        self.track_identities[matched.track_id] = temp_id
                        cv2.putText(frame, f"{temp_id} {score:.2f}", (x1, max(y1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                        body_box = (matched.bbox[0], matched.bbox[1], matched.bbox[2], matched.bbox[3])
                        recognized_faces[temp_id] = body_box
                    else:
                        cv2.putText(frame, f"Blurry {score:.2f}", (x1, max(y1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                else:
                    cv2.putText(frame, f"Unknown {score:.2f}", (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                continue
            
            cv2.putText(frame, f"{identity} {score:.2f}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            matched = self._match_face_to_track((x1, y1, x2, y2), tracks)
            
            if matched is not None:
                body_box = (matched.bbox[0], matched.bbox[1], matched.bbox[2], matched.bbox[3])
                previous_track = self.identity_claims.get(identity)
                if previous_track is not None and previous_track != matched.track_id:
                    self.track_identities.pop(previous_track, None)
                self.track_identities[matched.track_id] = identity
                self.identity_claims[identity] = matched.track_id
                track_id_str = f"[T{matched.track_id}]"
            else:
                body_box = self._face_to_body_bbox((x1, y1, x2, y2), frame.shape)
                track_id_str = ""
            
            recognized_faces[identity] = body_box
            cv2.rectangle(frame, (body_box[0], body_box[1]), (body_box[2], body_box[3]), (0, 255, 0), 3)
            # Show name with track ID for ReID tracking
            label_text = f"{identity} {track_id_str}" if track_id_str else identity
            cv2.putText(frame, label_text, (body_box[0], body_box[3] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            crop = self._crop(frame, (x1, y1, x2, y2))
            mask_probability = mask_classifier.probability(crop)
            mask_memory.update(identity, mask_probability, timestamp)

        # Clean up stale track identities
        active_ids = {track.track_id for track in tracks}
        self.track_identities = {tid: name for tid, name in self.track_identities.items() if tid in active_ids}
        self.identity_claims = {name: tid for name, tid in self.identity_claims.items() if tid in active_ids}

        # Draw unrecognized tracks and add them to named_boxes for collision detection
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            assigned = self.track_identities.get(track.track_id)
            
            if assigned is None:
                # Unrecognized person - try to get Unknown ID from unknown_tracker
                # Generate a consistent Unknown ID based on track
                unknown_id = f"Unknown_{track.track_id:03d}"
                color = (255, 191, 0)  # Cyan/Yellow for unrecognized
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, unknown_id, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                # Add to named_boxes so collision detection can use it
                if unknown_id not in recognized_faces:
                    recognized_faces[unknown_id] = (x1, y1, x2, y2)
        
        named_boxes = recognized_faces
        cv2.putText(frame, self.label, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        return frame, named_boxes

    @staticmethod
    def _crop(frame: np.ndarray, bbox: BBox) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=frame.dtype)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _face_to_body_bbox(face_bbox: BBox, frame_shape: Tuple[int, int, int]) -> BBox:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = face_bbox
        face_h = max(1, y2 - y1)
        pad_x = int(face_h * 0.8)
        pad_down = int(face_h * 3.0)
        bx1 = max(0, x1 - pad_x)
        bx2 = min(w - 1, x2 + pad_x)
        by2 = min(h - 1, y2 + pad_down)
        return (bx1, y1, bx2, by2)

    @staticmethod
    def _predict_identity(
        embedding: np.ndarray,
        names: np.ndarray,
        embeddings: np.ndarray,
        threshold: float,
    ) -> Tuple[str, float]:
        if embeddings.size == 0:
            return "Unknown", 0.0
        similarities = embeddings @ embedding
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        if best_score < threshold:
            return "Unknown", best_score
        return str(names[best_idx]), best_score

    @staticmethod
    def _match_face_to_track(face_bbox: BBox, tracks: Sequence[TrackInfo]) -> Optional[TrackInfo]:
        if not tracks:
            return None
        best_track: Optional[TrackInfo] = None
        best_iou = 0.0
        for track in tracks:
            iou = ViewPipeline._bbox_iou(face_bbox, track.bbox)
            if iou > best_iou:
                best_iou = iou
                best_track = track
        if best_track is not None and best_iou >= 0.05:
            return best_track
        fx1, fy1, fx2, fy2 = face_bbox
        cx = (fx1 + fx2) / 2
        cy = (fy1 + fy2) / 2
        fallback: Optional[TrackInfo] = None
        best_area = float("inf")
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best_area = area
                    fallback = track
        return fallback

    @staticmethod
    def _bbox_iou(box_a: BBox, box_b: BBox) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union


class ContactMonitorService:
    """Service class for contact monitoring that can be controlled from the web."""
    
    def __init__(
        self,
        mode: str = "video",
        front_video_path: Optional[str] = None,
        side_video_path: Optional[str] = None,
        front_camera_index: int = 0,
        side_camera_index: int = 1,
        use_gpu: bool = False,
        min_confidence: float = 0.35,
        threshold: float = 0.32,
        base_rate: float = 0.02,
        event_penalty: float = 0.05,
    ):
        self.mode = mode
        self.front_video_path = front_video_path
        self.side_video_path = side_video_path
        self.front_camera_index = front_camera_index
        self.side_camera_index = side_camera_index
        self.use_gpu = use_gpu
        self.min_confidence = min_confidence
        self.threshold = threshold
        self.base_rate = base_rate
        self.event_penalty = event_penalty
        
        # Will be initialized in start()
        self.front_view: Optional[ViewPipeline] = None
        self.side_view: Optional[ViewPipeline] = None
        self.front_capture: Optional[cv2.VideoCapture] = None
        self.side_capture: Optional[cv2.VideoCapture] = None
        self.analyzer = None
        self.names: Optional[np.ndarray] = None
        self.embeddings: Optional[np.ndarray] = None
        self.mask_classifier: Optional[MaskClassifier] = None
        self.mask_memory: Optional[MaskMemory] = None
        self.ledger: Optional[ContactLedger] = None
        self.email_alerter: Optional[EmailAlerter] = None
        self.unknown_tracker: Optional[UnknownPersonTracker] = None
        self.mdr_patients: set = set()
        self.mdr_alert_threshold_seconds: float = 300.0
        self.contact_log_min_duration: float = 3.0  # Log contacts after 3 seconds
        
        # Collision tracking
        self.front_collision_tracker: Optional[CollisionTracker] = None
        self.side_collision_tracker: Optional[CollisionTracker] = None
        self.alert_system: Optional[AlertSystem] = None
        
        # State
        self.pair_states: Dict[Tuple[str, str], PairState] = {}
        self.recent_front: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self.recent_side: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self.frame_number: int = 0
        self._initialized: bool = False
        
        # Store latest frames for MDR alert snapshots
        self._last_front_frame: Optional[np.ndarray] = None
        self._last_side_frame: Optional[np.ndarray] = None

    def initialize(self):
        """Initialize all components."""
        if self._initialized:
            return
        
        # Load face database
        registry = load_facebank()
        if not registry:
            # No registered persons - allow monitoring but all will be unknown
            print("[MonitorService] Warning: No embeddings found. All persons will be tracked as unknown.")
            self.names = np.array([], dtype=str)
            self.embeddings = np.zeros((0, 512), dtype=np.float32)
        else:
            self.names, self.embeddings = flatten_registry(registry)
            self.embeddings = self.embeddings.astype(np.float32)
        
        # Load MDR patients
        self.mdr_patients = load_mdr_patients()
        self.email_alerter = EmailAlerter()
        self.mdr_alert_threshold_seconds = float(os.getenv("MDR_ALERT_THRESHOLD_SECONDS", "300"))
        self.contact_log_min_duration = float(os.getenv("CONTACT_LOG_MIN_DURATION_SECONDS", "3"))
        
        # Initialize analyzer and classifiers
        det_size = recognition_settings.det_size
        self.analyzer = get_analyzer((det_size, det_size), use_gpu=self.use_gpu)
        self.mask_classifier = MaskClassifier()
        self.ledger = ContactLedger()
        self.mask_memory = MaskMemory(contact_settings.mask_decay_seconds)
        self.unknown_tracker = get_unknown_tracker()
        
        # Initialize collision tracking
        self.front_collision_tracker = CollisionTracker(collision_settings.alert_duration_seconds)
        self.side_collision_tracker = CollisionTracker(collision_settings.alert_duration_seconds)
        self.alert_system = AlertSystem(
            min_risk=collision_settings.min_risk_for_alert,
            duration_threshold=collision_settings.alert_duration_seconds,
            min_alert_interval=collision_settings.alert_cooldown_seconds,
            enable_logging=True,
            enable_audio=False,  # Disable audio for web
        )
        
        # Open video captures
        if self.mode == "video":
            print(f"[MonitorService] Opening video files: front={self.front_video_path}, side={self.side_video_path}")
            self.front_capture = cv2.VideoCapture(self.front_video_path)
            self.side_capture = cv2.VideoCapture(self.side_video_path)
        else:  # webcam mode
            print(f"[MonitorService] Opening webcams: front_idx={self.front_camera_index}, side_idx={self.side_camera_index}")
            self.front_capture = cv2.VideoCapture(self.front_camera_index)
            self.side_capture = cv2.VideoCapture(self.side_camera_index)
        
        print(f"[MonitorService] Front camera opened: {self.front_capture.isOpened()}")
        print(f"[MonitorService] Side camera opened: {self.side_capture.isOpened()}")
        
        if not self.front_capture.isOpened():
            raise ValueError(f"Cannot open front video/camera (index={self.front_camera_index})")
        if not self.side_capture.isOpened():
            raise ValueError(f"Cannot open side video/camera (index={self.side_camera_index})")
        
        # Create view pipelines
        self.front_view = ViewPipeline(
            label="Front",
            tracker=self._build_tracker(),
            capture=self.front_capture,
        )
        self.side_view = ViewPipeline(
            label="Side",
            tracker=self._build_tracker(),
            capture=self.side_capture,
        )
        
        self._initialized = True
        print(f"[MonitorService] Initialized - mode={self.mode}, mdr_patients={len(self.mdr_patients)}")

    def _build_tracker(self) -> PersonTracker:
        use_gpu = recognition_settings.reid_embedder_gpu or self.use_gpu
        return PersonTracker(
            model_path=str(recognition_settings.reid_model_path or "yolov8n.pt"),
            detection_confidence=recognition_settings.reid_detector_conf,
            embedder_gpu=use_gpu,
            nms_iou=recognition_settings.reid_nms_iou,
            box_shrink=recognition_settings.reid_box_shrink,
            device='cuda' if use_gpu else None,
        )

    def cleanup(self):
        """Release resources."""
        if self.front_capture:
            self.front_capture.release()
        if self.side_capture:
            self.side_capture.release()
        if self.unknown_tracker:
            self.unknown_tracker.flush_all()
        self._initialized = False
        print("[MonitorService] Cleaned up")

    async def run_async(self, stop_event: Optional[asyncio.Event] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the monitoring loop asynchronously, yielding frame data."""
        self.initialize()
        
        last_tick = time.monotonic()
        frame_interval = max(self.front_view.frame_interval, self.side_view.frame_interval)
        
        try:
            while True:
                # Check stop signal
                if stop_event and stop_event.is_set():
                    break
                
                # Read frames
                ok_front, frame_front = self.front_view.read()
                ok_side, frame_side = self.side_view.read()
                
                if not ok_front or not ok_side:
                    yield {
                        "status": "ended",
                        "message": "Video stream ended",
                        "frame_number": self.frame_number
                    }
                    break
                
                now = time.monotonic()
                elapsed = now - last_tick
                last_tick = now
                delta_t = frame_interval if frame_interval > 0 else max(elapsed, 1.0 / 30.0)
                
                # Process frames
                front_processed, front_boxes = self.front_view.process(
                    frame_front,
                    analyzer=self.analyzer,
                    names=self.names,
                    embeddings=self.embeddings,
                    min_confidence=self.min_confidence,
                    threshold=self.threshold,
                    mask_classifier=self.mask_classifier,
                    mask_memory=self.mask_memory,
                    timestamp=now,
                    unknown_tracker=self.unknown_tracker,
                )
                
                side_processed, side_boxes = self.side_view.process(
                    frame_side,
                    analyzer=self.analyzer,
                    names=self.names,
                    embeddings=self.embeddings,
                    min_confidence=self.min_confidence,
                    threshold=self.threshold,
                    mask_classifier=self.mask_classifier,
                    mask_memory=self.mask_memory,
                    timestamp=now,
                    unknown_tracker=self.unknown_tracker,
                )
                
                # Store latest processed frames for MDR alert snapshots
                self._last_front_frame = front_processed.copy()
                self._last_side_frame = side_processed.copy()
                
                # Detect collisions
                front_bboxes = self._to_bounding_boxes(front_boxes)
                side_bboxes = self._to_bounding_boxes(side_boxes)
                
                front_collisions = self.front_collision_tracker.update_collisions(
                    detect_collisions(
                        front_bboxes,
                        iou_threshold=collision_settings.iou_threshold,
                        distance_threshold=collision_settings.distance_threshold,
                        distance_meters_threshold=collision_settings.distance_meters_threshold,
                        frame_width=frame_front.shape[1],
                        frame_height=frame_front.shape[0],
                    )
                )
                
                side_collisions = self.side_collision_tracker.update_collisions(
                    detect_collisions(
                        side_bboxes,
                        iou_threshold=collision_settings.iou_threshold,
                        distance_threshold=collision_settings.distance_threshold,
                        distance_meters_threshold=collision_settings.distance_meters_threshold,
                        frame_width=frame_side.shape[1],
                        frame_height=frame_side.shape[0],
                    )
                )
                
                # Process contacts
                active_contacts = self._process_contacts(
                    front_collisions, side_collisions,
                    front_boxes, side_boxes,
                    front_processed, side_processed,
                    now, delta_t
                )
                
                # Draw distance lines between persons on each frame
                self._draw_collision_distances(
                    front_processed, front_collisions, front_boxes,
                    collision_settings.distance_meters_threshold
                )
                self._draw_collision_distances(
                    side_processed, side_collisions, side_boxes,
                    collision_settings.distance_meters_threshold
                )
                
                # Combine frames
                combined = self._combine_frames(front_processed, side_processed)
                self._draw_risk_overlay(combined)
                
                # Encode frame as base64 JPEG
                _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Yield frame data
                yield {
                    "status": "running",
                    "frame_number": self.frame_number,
                    "frame_base64": frame_base64,
                    "detected_persons": {
                        "front": list(front_boxes.keys()),
                        "side": list(side_boxes.keys()),
                    },
                    "active_contacts": active_contacts,
                    "stats": {
                        "total_contacts": len(self.pair_states),
                        "mdr_contacts": sum(1 for s in self.pair_states.values() if s.involves_mdr),
                        "unknown_contacts": sum(1 for s in self.pair_states.values() if s.involves_unknown),
                    }
                }
                
                self.frame_number += 1
                
                # Small delay for async yielding
                await asyncio.sleep(0.01)
                
        finally:
            self._flush_active_pairs()

    def _to_bounding_boxes(self, named_boxes: Dict[str, BBox]) -> List[BoundingBox]:
        boxes: List[BoundingBox] = []
        for name, (x1, y1, x2, y2) in named_boxes.items():
            boxes.append(BoundingBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2), person_name=name))
        return boxes

    def _combine_frames(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_h, left_w = left.shape[:2]
        right_h, right_w = right.shape[:2]
        target_height = max(left_h, right_h)
        
        def _resize(frame: np.ndarray) -> np.ndarray:
            if frame.shape[0] == target_height:
                return frame
            scale = target_height / frame.shape[0]
            new_width = int(frame.shape[1] * scale)
            return cv2.resize(frame, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        left_resized = _resize(left)
        right_resized = _resize(right)
        return np.hstack([left_resized, right_resized])

    def _draw_collision_distances(
        self,
        frame: np.ndarray,
        collisions: List[Collision],
        named_boxes: Dict[str, BBox],
        distance_threshold_meters: float = 1.5,
    ) -> np.ndarray:
        """Draw distance lines between persons when within threshold.
        
        Args:
            frame: The frame to draw on
            collisions: List of detected collisions with distance info
            named_boxes: Dict of person names to their bounding boxes
            distance_threshold_meters: Only show distance when closer than this (meters)
            
        Returns:
            Frame with distance visualizations drawn
        """
        for collision in collisions:
            # Get distance in meters (if available)
            distance_m = collision.distance_meters
            
            # Only draw if we have a valid distance and it's within threshold
            if distance_m is None:
                continue
            
            if distance_m > distance_threshold_meters:
                continue
            
            # Get bounding box centers
            box1 = collision.box1
            box2 = collision.box2
            
            center1 = (int((box1.x1 + box1.x2) / 2), int((box1.y1 + box1.y2) / 2))
            center2 = (int((box2.x1 + box2.x2) / 2), int((box2.y1 + box2.y2) / 2))
            
            # Determine color based on distance (closer = more red)
            if distance_m <= 0.5:
                color = (0, 0, 255)  # Red - very close
                thickness = 4
            elif distance_m <= 1.0:
                color = (0, 128, 255)  # Orange - close
                thickness = 3
            else:
                color = (0, 255, 255)  # Yellow - within threshold
                thickness = 2
            
            # Draw line connecting the two persons
            cv2.line(frame, center1, center2, color, thickness)
            
            # Calculate midpoint for distance label
            mid_x = int((center1[0] + center2[0]) / 2)
            mid_y = int((center1[1] + center2[1]) / 2)
            
            # Draw distance label with background
            distance_text = f"{distance_m:.2f}m"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            
            (text_w, text_h), baseline = cv2.getTextSize(distance_text, font, font_scale, font_thickness)
            
            # Background rectangle
            padding = 4
            cv2.rectangle(
                frame,
                (mid_x - text_w // 2 - padding, mid_y - text_h // 2 - padding),
                (mid_x + text_w // 2 + padding, mid_y + text_h // 2 + padding),
                color,
                -1  # Filled
            )
            
            # Text
            cv2.putText(
                frame,
                distance_text,
                (mid_x - text_w // 2, mid_y + text_h // 2 - 2),
                font,
                font_scale,
                (255, 255, 255),  # White text
                font_thickness,
            )
            
            # Draw person names at their positions if not already labeled
            for box, name in [(box1, collision.person1), (box2, collision.person2)]:
                # Draw small indicator near the person
                box_center_x = int((box.x1 + box.x2) / 2)
                box_bottom = box.y2 + 15
                
                # Small distance indicator under the person
                cv2.putText(
                    frame,
                    f"<{distance_m:.1f}m",
                    (box_center_x - 25, min(box_bottom + 20, frame.shape[0] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )
        
        return frame

    def _mask_modifier(self, prob_a: float, prob_b: float) -> float:
        effect = contact_settings.mask_effect
        modifier_a = max(0.05, 1.0 - prob_a * effect)
        modifier_b = max(0.05, 1.0 - prob_b * effect)
        return modifier_a * modifier_b

    def _pairs_from_collisions(self, collisions: Sequence[Collision], min_risk: float) -> Dict[Tuple[str, str], Tuple[float, Optional[float]]]:
        """Extract pairs from collisions with risk score and distance_meters.
        
        Returns:
            Dict mapping collision_id to (risk_score, distance_meters)
        """
        pairs: Dict[Tuple[str, str], Tuple[float, Optional[float]]] = {}
        for collision in collisions:
            if collision.risk_score < min_risk:
                continue
            pairs[collision.get_collision_id()] = (collision.risk_score, collision.distance_meters)
        return pairs

    def _update_recent_contacts(
        self,
        store: Dict[Tuple[str, str], Tuple[float, float, Optional[float]]],
        overlaps: Dict[Tuple[str, str], Tuple[float, Optional[float]]],
        timestamp: float,
        ttl: float,
    ) -> None:
        """Update recent contacts store with risk, timestamp, and distance.
        
        Args:
            store: Dict mapping pair to (risk_score, timestamp, distance_meters)
            overlaps: Dict mapping pair to (risk_score, distance_meters)
            timestamp: Current timestamp
            ttl: Time-to-live for entries
        """
        for pair, (risk, distance_m) in overlaps.items():
            store[pair] = (risk, timestamp, distance_m)
        if ttl <= 0:
            return
        stale = [pair for pair, (_, ts, _) in store.items() if timestamp - ts > ttl]
        for pair in stale:
            store.pop(pair, None)

    def _process_contacts(
        self,
        front_collisions: List[Collision],
        side_collisions: List[Collision],
        front_boxes: Dict[str, BBox],
        side_boxes: Dict[str, BBox],
        front_frame: np.ndarray,
        side_frame: np.ndarray,
        now: float,
        delta_t: float,
    ) -> List[Dict[str, Any]]:
        """Process contact detection and return list of active contacts."""
        overlap_threshold = contact_settings.overlap_threshold
        
        front_pairs = self._pairs_from_collisions(front_collisions, overlap_threshold)
        side_pairs = self._pairs_from_collisions(side_collisions, overlap_threshold)
        self._update_recent_contacts(self.recent_front, front_pairs, now, contact_settings.pair_sync_window)
        self._update_recent_contacts(self.recent_side, side_pairs, now, contact_settings.pair_sync_window)
        
        # Determine confirmed pairs
        if contact_settings.require_both_cameras:
            confirmed_pairs = set(self.recent_front.keys()).intersection(self.recent_side.keys())
        else:
            confirmed_pairs = set(self.recent_front.keys()).union(self.recent_side.keys())
        
        timestamp_iso = get_local_timestamp_iso()
        registered_names = set(self.names)
        active_contacts = []
        
        for pair in confirmed_pairs:
            state = self.pair_states.setdefault(pair, PairState())
            prob_a = self.mask_memory.probability(pair[0], now)
            prob_b = self.mask_memory.probability(pair[1], now)
            modifier = self._mask_modifier(prob_a, prob_b)
            delta_risk = self.base_rate * modifier * delta_t
            
            # Get distance_meters from recent contacts (prefer front camera if available)
            current_distance_m: Optional[float] = None
            if pair in self.recent_front:
                _, _, current_distance_m = self.recent_front[pair]
            elif pair in self.recent_side:
                _, _, current_distance_m = self.recent_side[pair]
            
            if not state.active:
                delta_risk += self.event_penalty
                state.active = True
                state.start_iso = timestamp_iso
                state.start_timestamp = time.time()
                state.person_a = pair[0]
                state.person_b = pair[1]
                state.min_distance_meters = current_distance_m
                
                # Load existing risks
                existing_risk_a, existing_risk_b = get_bidirectional_risks(pair[0], pair[1])
                state.cumulative_risk_a = existing_risk_a
                state.cumulative_risk_b = existing_risk_b
                state.cumulative = (existing_risk_a + existing_risk_b) / 2.0
                
                # Check MDR involvement
                if pair[0] in self.mdr_patients:
                    state.involves_mdr = True
                    state.mdr_patient = pair[0]
                    state.other_person = pair[1]
                    from mdr_tracker_mongo import get_pathogen_info
                    state.pathogen_type, state.pathogen_factor = get_pathogen_info(pair[0])
                elif pair[1] in self.mdr_patients:
                    state.involves_mdr = True
                    state.mdr_patient = pair[1]
                    state.other_person = pair[0]
                    from mdr_tracker_mongo import get_pathogen_info
                    state.pathogen_type, state.pathogen_factor = get_pathogen_info(pair[1])
                
                # Check unknown involvement
                if pair[0].startswith("Unknown_") and pair[1] in registered_names:
                    state.involves_unknown = True
                    state.unknown_temp_id = pair[0]
                elif pair[1].startswith("Unknown_") and pair[0] in registered_names:
                    state.involves_unknown = True
                    state.unknown_temp_id = pair[1]
            
            # Update current distance and track minimum
            state.last_distance_meters = current_distance_m
            if current_distance_m is not None:
                if state.min_distance_meters is None or current_distance_m < state.min_distance_meters:
                    state.min_distance_meters = current_distance_m
            
            # Update risks
            state.cumulative_risk_a += delta_risk
            state.cumulative_risk_b += delta_risk
            state.cumulative = (state.cumulative_risk_a + state.cumulative_risk_b) / 2.0
            state.end_iso = timestamp_iso
            
            # Update mask status
            state.mdr_patient_masked = prob_a > 0.5 if pair[0] == state.mdr_patient else prob_b > 0.5
            state.other_person_masked = prob_b > 0.5 if pair[0] == state.mdr_patient else prob_a > 0.5
            
            active_contacts.append({
                "person_a": pair[0],
                "person_b": pair[1],
                "risk_a": round(state.cumulative_risk_a * 100, 1),
                "risk_b": round(state.cumulative_risk_b * 100, 1),
                "involves_mdr": state.involves_mdr,
                "involves_unknown": state.involves_unknown,
                "duration_seconds": time.time() - state.start_timestamp,
                "distance_meters": round(current_distance_m, 2) if current_distance_m is not None else None,
                "min_distance_meters": round(state.min_distance_meters, 2) if state.min_distance_meters is not None else None,
            })
        
        # Handle inactive pairs
        inactive_pairs = set(self.pair_states.keys()) - confirmed_pairs
        for pair in list(inactive_pairs):
            state = self.pair_states[pair]
            if state.active and state.cumulative_risk_a > 0:
                contact_duration = time.time() - state.start_timestamp
                
                # Save risks to database
                if state.person_a and state.person_b:
                    update_bidirectional_risks(
                        state.person_a, state.person_b,
                        state.cumulative_risk_a, state.cumulative_risk_b,
                        contact_duration
                    )
                
                # Log ALL contacts to ledger if minimum duration threshold met
                if contact_duration >= self.contact_log_min_duration:
                    self.ledger.log_incident(
                        state.person_a, state.person_b,
                        start_time=state.start_iso,
                        end_time=state.end_iso,
                        cumulative_risk=state.cumulative_risk_a,
                        is_mdr_contact=state.involves_mdr,
                        mdr_risk_score=state.mdr_risk_score if state.involves_mdr else 0.0,
                        pathogen_type=state.pathogen_type,
                        pathogen_factor=state.pathogen_factor,
                        distance_meters=state.last_distance_meters,
                        min_distance_meters=state.min_distance_meters,
                    )
                    self.ledger.log_incident(
                        state.person_b, state.person_a,
                        start_time=state.start_iso,
                        end_time=state.end_iso,
                        cumulative_risk=state.cumulative_risk_b,
                        is_mdr_contact=state.involves_mdr,
                        mdr_risk_score=state.mdr_risk_score if state.involves_mdr else 0.0,
                        pathogen_type=state.pathogen_type,
                        pathogen_factor=state.pathogen_factor,
                        distance_meters=state.last_distance_meters,
                        min_distance_meters=state.min_distance_meters,
                    )
                    
                    # Send MDR alert if duration exceeds MDR threshold
                    if state.involves_mdr and contact_duration >= self.mdr_alert_threshold_seconds:
                        self._send_mdr_alert(state, contact_duration)
            
            self.pair_states.pop(pair, None)
        
        return active_contacts

    def _draw_risk_overlay(self, frame: np.ndarray) -> None:
        """Draw risk overlay on the combined frame."""
        entries = [
            (pair[0], pair[1], state.cumulative_risk_a, state.cumulative_risk_b, state.active, 
             state.involves_mdr, state.mdr_patient, state.pathogen_type)
            for pair, state in self.pair_states.items() if state.cumulative_risk_a > 0
        ]
        entries.sort(key=lambda item: (item[5], item[2] + item[3]), reverse=True)
        
        y_offset = 60
        
        for idx, (person_a, person_b, risk_a, risk_b, active, involves_mdr, mdr_patient, pathogen_type) in enumerate(entries[:6]):
            risk_percent_a = min(risk_a * 100, 100)
            risk_percent_b = min(risk_b * 100, 100)
            avg_risk = (risk_percent_a + risk_percent_b) / 2.0
            
            if involves_mdr:
                if avg_risk >= 60:
                    color = (0, 0, 255)  # Red
                else:
                    color = (0, 165, 255)  # Orange
                text = f"🚨 MDR [{pathogen_type}]: {person_a}:{risk_percent_a:.1f}% ↔ {person_b}:{risk_percent_b:.1f}%"
            elif avg_risk >= 40:
                color = (0, 165, 255)  # Orange
                text = f"⚠ HIGH: {person_a}:{risk_percent_a:.1f}% ↔ {person_b}:{risk_percent_b:.1f}%"
            elif active:
                color = (0, 255, 255)  # Yellow
                text = f"{person_a}:{risk_percent_a:.1f}% ↔ {person_b}:{risk_percent_b:.1f}%*"
            else:
                color = (200, 200, 200)  # Gray
                text = f"{person_a}:{risk_percent_a:.1f}% ↔ {person_b}:{risk_percent_b:.1f}%"
            
            cv2.putText(frame, text, (25, y_offset + idx * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if not entries:
            cv2.putText(frame, "No active contacts", (25, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    def _flush_active_pairs(self):
        """Save all active pairs before shutdown."""
        for pair, state in list(self.pair_states.items()):
            if state.active and state.cumulative_risk_a > 0:
                contact_duration = time.time() - state.start_timestamp if state.start_timestamp > 0 else 0.0
                
                if state.person_a and state.person_b:
                    update_bidirectional_risks(
                        state.person_a, state.person_b,
                        state.cumulative_risk_a, state.cumulative_risk_b,
                        contact_duration
                    )
                    
                    # Log contacts if min duration met
                    if contact_duration >= self.contact_log_min_duration:
                        self.ledger.log_incident(
                            state.person_a, state.person_b,
                            start_time=state.start_iso,
                            end_time=state.end_iso or get_local_timestamp_iso(),
                            cumulative_risk=state.cumulative_risk_a,
                            is_mdr_contact=state.involves_mdr,
                            mdr_risk_score=state.mdr_risk_score if state.involves_mdr else 0.0,
                            pathogen_type=state.pathogen_type,
                            pathogen_factor=state.pathogen_factor,
                            distance_meters=state.last_distance_meters,
                            min_distance_meters=state.min_distance_meters,
                        )
                        self.ledger.log_incident(
                            state.person_b, state.person_a,
                            start_time=state.start_iso,
                            end_time=state.end_iso or get_local_timestamp_iso(),
                            cumulative_risk=state.cumulative_risk_b,
                            is_mdr_contact=state.involves_mdr,
                            mdr_risk_score=state.mdr_risk_score if state.involves_mdr else 0.0,
                            pathogen_type=state.pathogen_type,
                            pathogen_factor=state.pathogen_factor,
                            distance_meters=state.last_distance_meters,
                            min_distance_meters=state.min_distance_meters,
                        )
                        
                        # Send MDR alert if threshold met
                        if state.involves_mdr and contact_duration >= self.mdr_alert_threshold_seconds:
                            self._send_mdr_alert(state, contact_duration)
            
            self.pair_states.pop(pair, None)

    def _send_mdr_alert(self, state: PairState, duration_seconds: float):
        """Send MDR contact alert via email and store in database."""
        if state.mdr_alert_sent:
            return
        
        # Calculate risk percentage
        risk_percent = min(100.0, max(state.cumulative_risk_a, state.cumulative_risk_b) * 100.0)
        
        # Capture snapshots from current frames
        front_snapshot = self._last_front_frame.copy() if self._last_front_frame is not None else None
        side_snapshot = self._last_side_frame.copy() if self._last_side_frame is not None else None
        
        # Create alert data
        alert = MDRContactAlert(
            mdr_patient=state.mdr_patient,
            contacted_person=state.other_person,
            contact_start=state.start_iso,
            contact_end=state.end_iso,
            duration_seconds=duration_seconds,
            risk_percent=risk_percent,
            front_snapshot=front_snapshot,
            side_snapshot=side_snapshot,
            distance_meters=state.last_distance_meters,
            min_distance_meters=state.min_distance_meters,
        )
        
        # Send alert (stores in MongoDB and sends email)
        doc_id = self.email_alerter.send_mdr_alert(alert)
        
        if doc_id:
            state.mdr_alert_sent = True
            distance_str = f", distance={state.min_distance_meters:.2f}m" if state.min_distance_meters is not None else ""
            print(f"[MDR Alert] Sent alert for {state.mdr_patient} ↔ {state.other_person}, duration={duration_seconds:.1f}s, risk={risk_percent:.1f}%{distance_str}")
