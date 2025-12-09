"""Unknown person tracking with MongoDB storage.

This module handles:
1. Assigning temporary IDs to unregistered persons
2. Storing face snapshots of unknown persons
3. Tracking contacts between unknown and registered persons
4. MDR alerts for unknown person contacts
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np
from rich import print as rprint

from database import (
    get_unknown_persons_collection,
    get_unknown_contacts_collection,
    get_alerts_collection,
    generate_person_id,
)


# Minimum face quality thresholds
MIN_FACE_BLUR_THRESHOLD = 100.0  # Laplacian variance threshold (higher = sharper)
MIN_FACE_SIZE = 40  # Minimum face size in pixels (width or height)
MIN_FACE_DETECTION_SCORE = 0.5  # Minimum detection confidence
MIN_EMBEDDING_NORM = 0.3  # Minimum embedding vector norm (low = poor face quality)
MAX_YAW_ANGLE = 45.0  # Maximum yaw angle in degrees (side profile threshold)
MAX_PITCH_ANGLE = 35.0  # Maximum pitch angle in degrees (looking up/down threshold)
MIN_FACE_SYMMETRY = 0.4  # Minimum face symmetry ratio (1.0 = perfectly symmetric)
MIN_LANDMARK_CONFIDENCE = 0.6  # Minimum landmark detection confidence


def calculate_blur_score(image: np.ndarray) -> float:
    """Calculate blur score using Laplacian variance.
    
    Higher score = sharper/clearer image
    Lower score = blurrier image
    
    Args:
        image: BGR or grayscale image
        
    Returns:
        Laplacian variance (blur score). Higher is better.
    """
    if image is None or image.size == 0:
        return 0.0
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    return float(variance)


def estimate_face_pose_from_landmarks(landmarks: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
    """Estimate face pose (yaw, pitch, roll) from 5-point facial landmarks.
    
    InsightFace provides 5 landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
    
    Args:
        landmarks: 5x2 array of facial landmark coordinates
        face_bbox: Face bounding box (x1, y1, x2, y2)
        
    Returns:
        Tuple of (yaw, pitch, roll) angles in degrees
    """
    if landmarks is None or len(landmarks) < 5:
        return 0.0, 0.0, 0.0
    
    x1, y1, x2, y2 = face_bbox
    face_width = x2 - x1
    face_height = y2 - y1
    
    if face_width <= 0 or face_height <= 0:
        return 0.0, 0.0, 0.0
    
    # Extract landmark points
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]
    
    # Calculate eye center
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    
    # Calculate mouth center
    mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
    mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
    
    # Calculate face center
    face_center_x = (x1 + x2) / 2
    face_center_y = (y1 + y2) / 2
    
    # YAW: Estimate based on nose position relative to eye center
    # If nose is to the left/right of eye center, face is turned
    eye_distance = np.linalg.norm(right_eye - left_eye)
    if eye_distance > 0:
        nose_offset = (nose[0] - eye_center_x) / eye_distance
        # Map nose offset to approximate yaw angle
        yaw = np.degrees(np.arcsin(np.clip(nose_offset * 0.5, -1.0, 1.0))) * 2
    else:
        yaw = 0.0
    
    # PITCH: Estimate based on nose position relative to eye-mouth line
    vertical_distance = mouth_center_y - eye_center_y
    if vertical_distance > 0:
        expected_nose_y = eye_center_y + vertical_distance * 0.4
        nose_y_offset = (nose[1] - expected_nose_y) / vertical_distance
        pitch = np.degrees(np.arctan(nose_y_offset)) * 1.5
    else:
        pitch = 0.0
    
    # ROLL: Estimate based on eye line angle
    eye_vector = right_eye - left_eye
    roll = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
    
    return float(yaw), float(pitch), float(roll)


def calculate_face_symmetry(landmarks: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> float:
    """Calculate face symmetry score from landmarks.
    
    Symmetric faces (frontal view) should have eyes and mouth corners 
    equidistant from the center vertical line.
    
    Args:
        landmarks: 5x2 array of facial landmark coordinates
        face_bbox: Face bounding box (x1, y1, x2, y2)
        
    Returns:
        Symmetry score between 0.0 (asymmetric) and 1.0 (perfectly symmetric)
    """
    if landmarks is None or len(landmarks) < 5:
        return 0.0
    
    x1, y1, x2, y2 = face_bbox
    face_center_x = (x1 + x2) / 2
    face_width = x2 - x1
    
    if face_width <= 0:
        return 0.0
    
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]
    
    # Calculate distances from center for each pair
    left_eye_dist = abs(left_eye[0] - face_center_x)
    right_eye_dist = abs(right_eye[0] - face_center_x)
    left_mouth_dist = abs(left_mouth[0] - face_center_x)
    right_mouth_dist = abs(right_mouth[0] - face_center_x)
    
    # Calculate symmetry ratios (1.0 = perfectly symmetric)
    eye_symmetry = min(left_eye_dist, right_eye_dist) / max(left_eye_dist, right_eye_dist) if max(left_eye_dist, right_eye_dist) > 0 else 0
    mouth_symmetry = min(left_mouth_dist, right_mouth_dist) / max(left_mouth_dist, right_mouth_dist) if max(left_mouth_dist, right_mouth_dist) > 0 else 0
    
    # Average symmetry score
    symmetry = (eye_symmetry + mouth_symmetry) / 2
    
    return float(symmetry)


def are_landmarks_valid(landmarks: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Tuple[bool, str]:
    """Check if facial landmarks indicate a properly visible face.
    
    This function validates that:
    1. All 5 landmarks are present and within the face bounding box
    2. Landmarks are in expected relative positions (eyes above nose, nose above mouth)
    3. Eye distance is reasonable relative to face width
    
    Args:
        landmarks: 5x2 array of facial landmark coordinates
        face_bbox: Face bounding box (x1, y1, x2, y2)
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if landmarks is None:
        return False, "no_landmarks"
    
    if len(landmarks) < 5:
        return False, f"insufficient_landmarks:{len(landmarks)}"
    
    x1, y1, x2, y2 = face_bbox
    face_width = x2 - x1
    face_height = y2 - y1
    
    if face_width <= 0 or face_height <= 0:
        return False, "invalid_bbox"
    
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]
    
    # Expand bbox slightly for landmark check (landmarks can be slightly outside)
    margin = max(face_width, face_height) * 0.1
    x1_m, y1_m = x1 - margin, y1 - margin
    x2_m, y2_m = x2 + margin, y2 + margin
    
    # Check if all landmarks are within/near the bounding box
    for i, (lx, ly) in enumerate([left_eye, right_eye, nose, left_mouth, right_mouth]):
        if not (x1_m <= lx <= x2_m and y1_m <= ly <= y2_m):
            landmark_names = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]
            return False, f"landmark_outside_bbox:{landmark_names[i]}"
    
    # Check relative positions
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
    
    # Eyes should be above nose, nose above mouth
    if not (eye_center_y < nose[1] < mouth_center_y):
        return False, "invalid_landmark_order"
    
    # Eye distance should be reasonable (not too small indicating side profile)
    eye_distance = np.linalg.norm(right_eye - left_eye)
    min_eye_distance = face_width * 0.15  # At least 15% of face width
    if eye_distance < min_eye_distance:
        return False, f"eyes_too_close:{eye_distance:.1f}<{min_eye_distance:.1f}"
    
    return True, "landmarks_valid"


def is_embedding_quality_acceptable(embedding: Optional[np.ndarray], min_norm: float = MIN_EMBEDDING_NORM) -> Tuple[bool, str]:
    """Check if face embedding indicates good face quality.
    
    Poor quality faces (blurry, occluded, side profiles) often produce
    embeddings with low norms or high variance.
    
    Args:
        embedding: Face embedding vector (512-dimensional)
        min_norm: Minimum acceptable embedding norm
        
    Returns:
        Tuple of (is_acceptable, reason)
    """
    if embedding is None:
        return False, "no_embedding"
    
    # Check embedding norm (should be normalized to ~1.0, but quality affects this)
    norm = np.linalg.norm(embedding)
    if norm < min_norm:
        return False, f"low_embedding_norm:{norm:.3f}<{min_norm}"
    
    return True, f"embedding_ok:norm={norm:.3f}"


def is_face_quality_acceptable(
    face_crop: Optional[np.ndarray],
    face_score: float,
    face_landmarks: Optional[np.ndarray] = None,
    face_bbox: Optional[Tuple[int, int, int, int]] = None,
    face_embedding: Optional[np.ndarray] = None,
    min_blur_threshold: float = MIN_FACE_BLUR_THRESHOLD,
    min_face_size: int = MIN_FACE_SIZE,
    min_detection_score: float = MIN_FACE_DETECTION_SCORE,
    max_yaw: float = MAX_YAW_ANGLE,
    max_pitch: float = MAX_PITCH_ANGLE,
    min_symmetry: float = MIN_FACE_SYMMETRY,
) -> Tuple[bool, str]:
    """Check if face quality is acceptable for unknown person registration.
    
    This comprehensive check ensures only clear, frontal faces are stored:
    1. Face detection confidence score
    2. Minimum face size in pixels
    3. Image blur (Laplacian variance)
    4. Face pose (yaw/pitch angles) - rejects side profiles
    5. Face symmetry - ensures frontal view
    6. Landmark validity - ensures proper face structure
    7. Embedding quality - validates face recognition quality
    
    Args:
        face_crop: Face image crop
        face_score: Face detection confidence score
        face_landmarks: 5-point facial landmarks from InsightFace (optional)
        face_bbox: Face bounding box (x1, y1, x2, y2) (optional)
        face_embedding: Face embedding vector (optional)
        min_blur_threshold: Minimum blur score (Laplacian variance)
        min_face_size: Minimum face size in pixels
        min_detection_score: Minimum detection confidence
        max_yaw: Maximum yaw angle (side-to-side)
        max_pitch: Maximum pitch angle (up-down)
        min_symmetry: Minimum face symmetry ratio
        
    Returns:
        Tuple of (is_acceptable, reason)
    """
    if face_crop is None or face_crop.size == 0:
        return False, "no_face_crop"
    
    # Check detection score
    if face_score < min_detection_score:
        return False, f"low_detection_score:{face_score:.2f}<{min_detection_score}"
    
    # Check face size
    h, w = face_crop.shape[:2]
    if w < min_face_size or h < min_face_size:
        return False, f"face_too_small:{w}x{h}<{min_face_size}"
    
    # Check blur score
    blur_score = calculate_blur_score(face_crop)
    if blur_score < min_blur_threshold:
        return False, f"too_blurry:{blur_score:.1f}<{min_blur_threshold}"
    
    # If landmarks and bbox are provided, perform additional checks
    if face_landmarks is not None and face_bbox is not None:
        # Validate landmarks
        landmarks_valid, landmarks_reason = are_landmarks_valid(face_landmarks, face_bbox)
        if not landmarks_valid:
            return False, f"invalid_face_structure:{landmarks_reason}"
        
        # Estimate face pose
        yaw, pitch, roll = estimate_face_pose_from_landmarks(face_landmarks, face_bbox)
        
        # Check yaw angle (side profile)
        if abs(yaw) > max_yaw:
            return False, f"side_profile:yaw={yaw:.1f}>{max_yaw}"
        
        # Check pitch angle (looking up/down)
        if abs(pitch) > max_pitch:
            return False, f"extreme_pitch:{pitch:.1f}>{max_pitch}"
        
        # Check face symmetry
        symmetry = calculate_face_symmetry(face_landmarks, face_bbox)
        if symmetry < min_symmetry:
            return False, f"asymmetric_face:{symmetry:.2f}<{min_symmetry}"
    
    # Check embedding quality if provided
    if face_embedding is not None:
        embedding_ok, embedding_reason = is_embedding_quality_acceptable(face_embedding)
        if not embedding_ok:
            return False, embedding_reason
    
    return True, f"ok:blur={blur_score:.1f},size={w}x{h},score={face_score:.2f}"


@dataclass
class UnknownPersonState:
    """Track state of an unknown person during monitoring session."""
    temp_id: str
    first_seen: float  # Unix timestamp (time.time())
    last_seen: float  # Unix timestamp (time.time())
    first_seen_monotonic: float = 0.0  # monotonic timestamp for duration calculations
    last_seen_monotonic: float = 0.0  # monotonic timestamp for duration calculations
    best_face_snapshot: Optional[np.ndarray] = None
    best_face_score: float = 0.0
    face_embedding: Optional[np.ndarray] = None
    track_id: Optional[int] = None
    active_contacts: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # registered_person -> (unix_time, monotonic_time)
    stored_in_db: bool = False
    mongodb_id: Optional[str] = None


class UnknownPersonTracker:
    """Track and manage unknown persons during contact monitoring."""
    
    def __init__(self):
        self.unknown_persons_col = get_unknown_persons_collection()
        self.unknown_contacts_col = get_unknown_contacts_collection()
        self.alerts_col = get_alerts_collection()
        
        # In-memory tracking during session
        self._next_temp_id = self._get_next_temp_id()
        self._active_unknowns: Dict[int, UnknownPersonState] = {}  # track_id -> state
        self._embedding_to_track: Dict[str, int] = {}  # embedding hash -> track_id
        
    def _get_next_temp_id(self) -> int:
        """Get next available temporary ID number from database."""
        latest = self.unknown_persons_col.find_one(
            {},
            sort=[("temp_id_num", -1)]
        )
        if latest and "temp_id_num" in latest:
            return latest["temp_id_num"] + 1
        return 1
    
    def _generate_temp_id(self) -> str:
        """Generate a new temporary ID."""
        temp_id = f"Unknown_{self._next_temp_id:04d}"
        self._next_temp_id += 1
        return temp_id
    
    def _embedding_hash(self, embedding: np.ndarray) -> str:
        """Create a hash from embedding for quick lookup."""
        return str(hash(embedding.tobytes()))
    
    def _match_to_existing_unknown(
        self, 
        embedding: np.ndarray,
        threshold: float = 0.5
    ) -> Optional[int]:
        """Try to match embedding to an existing unknown person by similarity."""
        if embedding is None:
            return None
        
        best_match_track_id = None
        best_similarity = threshold
        
        for track_id, state in self._active_unknowns.items():
            if state.face_embedding is not None:
                similarity = float(np.dot(embedding, state.face_embedding))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_track_id = track_id
        
        return best_match_track_id
    
    def register_unknown(
        self,
        track_id: int,
        face_crop: Optional[np.ndarray],
        face_score: float,
        face_embedding: Optional[np.ndarray],
        monotonic_timestamp: float,
        face_landmarks: Optional[np.ndarray] = None,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[UnknownPersonState]:
        """Register or update an unknown person.
        
        Only registers unknown persons with clear, frontal, non-blurry face images.
        Rejects side profiles, back of heads, occluded faces, and blurry images.
        
        Args:
            track_id: Track ID from object tracker
            face_crop: Face image crop
            face_score: Face detection confidence score
            face_embedding: Face embedding vector
            monotonic_timestamp: time.monotonic() value for duration calculations
            face_landmarks: 5-point facial landmarks from InsightFace (optional)
            face_bbox: Face bounding box (x1, y1, x2, y2) for landmark validation (optional)
            
        Returns:
            UnknownPersonState if registered/updated, None if face quality is too low
        """
        unix_now = time.time()  # Always use current Unix time for storage
        
        # Check if this track_id already exists
        if track_id in self._active_unknowns:
            state = self._active_unknowns[track_id]
            state.last_seen = unix_now
            state.last_seen_monotonic = monotonic_timestamp
            
            # Update face snapshot if better quality AND acceptable quality
            if face_crop is not None and face_score > state.best_face_score:
                is_acceptable, reason = is_face_quality_acceptable(
                    face_crop, face_score,
                    face_landmarks=face_landmarks,
                    face_bbox=face_bbox,
                    face_embedding=face_embedding,
                )
                if is_acceptable:
                    state.best_face_snapshot = face_crop.copy()
                    state.best_face_score = face_score
                    if face_embedding is not None:
                        state.face_embedding = face_embedding.copy()
            
            return state
        
        # Check face quality BEFORE creating new unknown (with full validation)
        is_acceptable, reason = is_face_quality_acceptable(
            face_crop, face_score,
            face_landmarks=face_landmarks,
            face_bbox=face_bbox,
            face_embedding=face_embedding,
        )
        if not is_acceptable:
            # Face quality too low - don't register as unknown
            # Log rejection reason for debugging
            rprint(f"[dim]Face rejected:[/] {reason}")
            return None
        
        # Try to match to existing unknown by embedding
        if face_embedding is not None:
            existing_track = self._match_to_existing_unknown(face_embedding)
            if existing_track is not None:
                state = self._active_unknowns[existing_track]
                state.track_id = track_id
                state.last_seen = unix_now
                state.last_seen_monotonic = monotonic_timestamp
                
                # Update face snapshot if better quality
                if face_crop is not None and face_score > state.best_face_score:
                    state.best_face_snapshot = face_crop.copy()
                    state.best_face_score = face_score
                    state.face_embedding = face_embedding.copy()
                
                # Re-register under new track_id
                del self._active_unknowns[existing_track]
                self._active_unknowns[track_id] = state
                return state
        
        # Create new unknown person (face quality already verified above)
        temp_id = self._generate_temp_id()
        blur_score = calculate_blur_score(face_crop)
        state = UnknownPersonState(
            temp_id=temp_id,
            first_seen=unix_now,
            last_seen=unix_now,
            first_seen_monotonic=monotonic_timestamp,
            last_seen_monotonic=monotonic_timestamp,
            best_face_snapshot=face_crop.copy() if face_crop is not None else None,
            best_face_score=face_score,
            face_embedding=face_embedding.copy() if face_embedding is not None else None,
            track_id=track_id,
        )
        self._active_unknowns[track_id] = state
        
        rprint(f"[cyan]New unknown person detected:[/] {temp_id} (blur={blur_score:.1f}, score={face_score:.2f})")
        return state
    
    def get_unknown_by_track(self, track_id: int) -> Optional[UnknownPersonState]:
        """Get unknown person state by track ID."""
        return self._active_unknowns.get(track_id)
    
    def get_temp_id_for_track(self, track_id: int) -> Optional[str]:
        """Get temporary ID for a track."""
        state = self._active_unknowns.get(track_id)
        return state.temp_id if state else None
    
    def log_contact_start(
        self,
        unknown_track_id: int,
        registered_person: str,
        monotonic_timestamp: float,
    ) -> None:
        """Log start of contact between unknown and registered person."""
        state = self._active_unknowns.get(unknown_track_id)
        if state is None:
            return
        
        if registered_person not in state.active_contacts:
            unix_now = time.time()
            state.active_contacts[registered_person] = (unix_now, monotonic_timestamp)
            rprint(f"[yellow]Contact started:[/] {state.temp_id} ↔ {registered_person}")
    
    def log_contact_end(
        self,
        unknown_track_id: int,
        registered_person: str,
        cumulative_risk: float,
        end_monotonic_timestamp: float,
        front_snapshot: Optional[np.ndarray] = None,
        side_snapshot: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Log end of contact and store in database. Returns document ID."""
        state = self._active_unknowns.get(unknown_track_id)
        if state is None:
            return None
        
        contact_times = state.active_contacts.pop(registered_person, None)
        if contact_times is None:
            return None
        
        start_unix, start_monotonic = contact_times
        end_unix = time.time()
        duration_seconds = end_monotonic_timestamp - start_monotonic
        
        # Ensure unknown person is stored in DB
        self._ensure_stored(state)
        
        # Create contact record (convert numpy types to Python native types)
        doc = {
            "unknown_temp_id": state.temp_id,
            "unknown_mongodb_id": state.mongodb_id,
            "registered_person": registered_person,
            "start_time": datetime.utcfromtimestamp(start_unix).isoformat(),
            "end_time": datetime.utcfromtimestamp(end_unix).isoformat(),
            "duration_seconds": float(duration_seconds),
            "cumulative_risk": float(cumulative_risk),
            "risk_percent": float(min(100.0, cumulative_risk * 100.0)),
            "timestamp": datetime.utcnow(),
            "created_at": datetime.utcnow(),
        }
        
        # Add snapshots if available
        if front_snapshot is not None:
            success, buffer = cv2.imencode('.jpg', front_snapshot)
            if success:
                doc["front_snapshot_base64"] = base64.b64encode(buffer).decode('utf-8')
        
        if side_snapshot is not None:
            success, buffer = cv2.imencode('.jpg', side_snapshot)
            if success:
                doc["side_snapshot_base64"] = base64.b64encode(buffer).decode('utf-8')
        
        result = self.unknown_contacts_col.insert_one(doc)
        
        rprint(f"[green]Contact logged:[/] {state.temp_id} ↔ {registered_person} "
               f"(duration: {duration_seconds:.1f}s, risk: {cumulative_risk*100:.1f}%)")
        
        return str(result.inserted_id)
    
    def _ensure_stored(self, state: UnknownPersonState) -> None:
        """Ensure unknown person is stored in database."""
        if state.stored_in_db:
            return
        
        doc = {
            "temp_id": state.temp_id,
            "temp_id_num": int(state.temp_id.split("_")[1]),
            "first_seen": datetime.utcfromtimestamp(state.first_seen),
            "last_seen": datetime.utcfromtimestamp(state.last_seen),
            "best_face_score": float(state.best_face_score),  # Convert numpy float to Python float
            "created_at": datetime.utcnow(),
        }
        
        # Store face snapshot
        if state.best_face_snapshot is not None:
            success, buffer = cv2.imencode('.jpg', state.best_face_snapshot)
            if success:
                doc["face_snapshot_base64"] = base64.b64encode(buffer).decode('utf-8')
        
        # Store face embedding for future matching (convert numpy array to list of Python floats)
        if state.face_embedding is not None:
            doc["face_embedding"] = [float(x) for x in state.face_embedding]
        
        result = self.unknown_persons_col.insert_one(doc)
        state.stored_in_db = True
        state.mongodb_id = str(result.inserted_id)
        
        rprint(f"[green]Unknown person stored:[/] {state.temp_id}")
    
    def send_mdr_alert_for_unknown(
        self,
        unknown_track_id: int,
        mdr_patient: str,
        duration_seconds: float,
        risk_percent: float,
        front_snapshot: Optional[np.ndarray] = None,
        side_snapshot: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Send MDR alert for unknown person contact. Returns alert document ID."""
        state = self._active_unknowns.get(unknown_track_id)
        if state is None:
            return None
        
        # Ensure unknown person is stored
        self._ensure_stored(state)
        
        # Convert numpy types to Python native types for MongoDB
        doc = {
            "alert_type": "mdr_contact_unknown",
            "mdr_patient": mdr_patient,
            "contacted_person": state.temp_id,
            "unknown_person_id": state.mongodb_id,
            "is_unknown_person": True,
            "duration_seconds": float(duration_seconds),
            "risk_percent": float(risk_percent),
            "contact_start": datetime.utcfromtimestamp(
                state.active_contacts.get(mdr_patient, (time.time(), 0))[0]
            ).isoformat(),
            "contact_end": None,  # Still ongoing
            "status": "ongoing",
            "created_at": datetime.utcnow(),
            "read": False,
            "email_sent": False,
        }
        
        # Add unknown person's face snapshot
        if state.best_face_snapshot is not None:
            success, buffer = cv2.imencode('.jpg', state.best_face_snapshot)
            if success:
                doc["unknown_face_base64"] = base64.b64encode(buffer).decode('utf-8')
        
        # Add scene snapshots
        if front_snapshot is not None:
            success, buffer = cv2.imencode('.jpg', front_snapshot)
            if success:
                doc["front_snapshot_base64"] = base64.b64encode(buffer).decode('utf-8')
        
        if side_snapshot is not None:
            success, buffer = cv2.imencode('.jpg', side_snapshot)
            if success:
                doc["side_snapshot_base64"] = base64.b64encode(buffer).decode('utf-8')
        
        result = self.alerts_col.insert_one(doc)
        
        rprint(f"[bold red]🚨 MDR ALERT (Unknown Person):[/] {state.temp_id} in contact with MDR patient {mdr_patient}")
        
        return str(result.inserted_id)
    
    def cleanup_stale_unknowns(self, max_age_seconds: float = 300.0) -> None:
        """Clean up unknown persons not seen recently."""
        current_time = time.time()
        stale_tracks = [
            track_id for track_id, state in self._active_unknowns.items()
            if current_time - state.last_seen > max_age_seconds
        ]
        
        for track_id in stale_tracks:
            state = self._active_unknowns.pop(track_id)
            # Store if they had any contacts
            if state.stored_in_db:
                # Update last_seen in database
                from bson import ObjectId
                self.unknown_persons_col.update_one(
                    {"_id": ObjectId(state.mongodb_id)},
                    {"$set": {"last_seen": datetime.utcfromtimestamp(state.last_seen)}}
                )
    
    def flush_all(self) -> None:
        """Store all active unknown persons and their pending contacts."""
        for track_id, state in list(self._active_unknowns.items()):
            if state.best_face_snapshot is not None or state.active_contacts:
                self._ensure_stored(state)


# Singleton instance
_tracker: Optional[UnknownPersonTracker] = None


def get_unknown_tracker() -> UnknownPersonTracker:
    """Get singleton unknown person tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = UnknownPersonTracker()
    return _tracker


# Query functions for API
def get_all_unknown_persons(limit: int = 100) -> List[Dict[str, Any]]:
    """Get all unknown persons from database."""
    col = get_unknown_persons_collection()
    contacts_col = get_unknown_contacts_collection()
    results = []
    
    # Get all MDR patient names for checking contacts
    from mdr_tracker_mongo import get_mdr_patients
    mdr_patients = set(get_mdr_patients())
    
    for doc in col.find().sort("created_at", -1).limit(limit):
        temp_id = doc["temp_id"]
        
        # Get contact count for this unknown person
        contact_count = contacts_col.count_documents({"unknown_temp_id": temp_id})
        
        # Check if they contacted any MDR patient
        contacted_mdr = contacts_col.count_documents({
            "unknown_temp_id": temp_id,
            "registered_person": {"$in": list(mdr_patients)}
        }) > 0
        
        results.append({
            "id": str(doc["_id"]),
            "temp_id": doc["temp_id"],
            "first_seen": doc["first_seen"],
            "last_seen": doc["last_seen"],
            "snapshot": doc.get("face_snapshot_base64"),  # Frontend expects 'snapshot'
            "contact_count": contact_count,
            "contacted_mdr": contacted_mdr,
            "best_face_score": doc.get("best_face_score", 0),
            "created_at": doc["created_at"],
        })
    return results


def get_unknown_person_detail(temp_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed info for an unknown person including face snapshot."""
    col = get_unknown_persons_collection()
    contacts_col = get_unknown_contacts_collection()
    doc = col.find_one({"temp_id": temp_id})
    if not doc:
        return None
    
    # Get contact count
    contact_count = contacts_col.count_documents({"unknown_temp_id": temp_id})
    
    return {
        "id": str(doc["_id"]),
        "temp_id": doc["temp_id"],
        "first_seen": doc["first_seen"],
        "last_seen": doc["last_seen"],
        "snapshot": doc.get("face_snapshot_base64"),
        "face_snapshot_base64": doc.get("face_snapshot_base64"),
        "contact_count": contact_count,
        "best_face_score": doc.get("best_face_score", 0),
        "created_at": doc["created_at"],
        "linked_to": doc.get("linked_to"),  # If linked to a registered person
    }


def mark_unknown_as_known(temp_id: str, person_name: str) -> bool:
    """Link an unknown person to a registered person.
    
    This updates the contact history to reference the registered person,
    then DELETES the unknown person record.
    """
    from database import get_persons_collection
    
    col = get_unknown_persons_collection()
    contacts_col = get_unknown_contacts_collection()
    persons_col = get_persons_collection()
    
    # Verify unknown person exists
    unknown_doc = col.find_one({"temp_id": temp_id})
    if not unknown_doc:
        return False
    
    # Verify registered person exists
    person_doc = persons_col.find_one({"name": person_name})
    if not person_doc:
        return False
    
    # Update all contacts to reference the linked person
    contacts_col.update_many(
        {"unknown_temp_id": temp_id},
        {"$set": {
            "linked_to_registered": person_name,
            "linked_at": datetime.utcnow(),
            "original_unknown_id": temp_id,
        }}
    )
    
    # DELETE the unknown person record (image is no longer needed since linked to existing person)
    col.delete_one({"temp_id": temp_id})
    
    rprint(f"[green]Unknown person linked and removed:[/] {temp_id} → {person_name}")
    return True


def get_unknown_contacts_for_person(registered_person: str) -> List[Dict[str, Any]]:
    """Get all unknown person contacts for a registered person."""
    col = get_unknown_contacts_collection()
    results = []
    
    # Aggregate contacts by unknown person
    pipeline = [
        {"$match": {"registered_person": registered_person}},
        {"$group": {
            "_id": "$unknown_temp_id",
            "contact_count": {"$sum": 1},
            "total_duration": {"$sum": "$duration_seconds"},
            "max_risk": {"$max": "$risk_percent"},
            "first_contact": {"$min": "$timestamp"},
            "last_contact": {"$max": "$timestamp"},
            "unknown_mongodb_id": {"$first": "$unknown_mongodb_id"},
        }},
        {"$sort": {"last_contact": -1}}
    ]
    
    for doc in col.aggregate(pipeline):
        results.append({
            "unknown_temp_id": doc["_id"],
            "unknown_id": doc.get("unknown_mongodb_id"),
            "contact_count": doc["contact_count"],
            "total_duration_seconds": doc["total_duration"],
            "max_risk_percent": doc["max_risk"],
            "first_contact": doc["first_contact"],
            "last_contact": doc["last_contact"],
        })
    
    return results


def get_unknown_contacts_with_mdr_patient(mdr_patient: str) -> List[Dict[str, Any]]:
    """Get all unknown persons who had contact with an MDR patient."""
    col = get_unknown_contacts_collection()
    unknown_col = get_unknown_persons_collection()
    results = []
    
    # Aggregate contacts by unknown person
    pipeline = [
        {"$match": {"registered_person": mdr_patient}},
        {"$group": {
            "_id": "$unknown_temp_id",
            "contact_count": {"$sum": 1},
            "total_duration": {"$sum": "$duration_seconds"},
            "max_risk": {"$max": "$risk_percent"},
            "first_contact": {"$min": "$timestamp"},
            "last_contact": {"$max": "$timestamp"},
            "unknown_mongodb_id": {"$first": "$unknown_mongodb_id"},
        }},
        {"$sort": {"max_risk": -1}}
    ]
    
    for doc in col.aggregate(pipeline):
        # Get face snapshot from unknown_persons collection
        unknown_doc = unknown_col.find_one({"temp_id": doc["_id"]})
        
        results.append({
            "unknown_temp_id": doc["_id"],
            "unknown_id": doc.get("unknown_mongodb_id"),
            "contact_count": doc["contact_count"],
            "total_duration_seconds": round(doc["total_duration"], 1),
            "max_risk_percent": round(doc["max_risk"], 1),
            "first_contact": doc["first_contact"],
            "last_contact": doc["last_contact"],
            "face_snapshot_base64": unknown_doc.get("face_snapshot_base64") if unknown_doc else None,
        })
    
    return results


def delete_unknown_person(temp_id: str) -> bool:
    """Delete an unknown person and their contact history.
    
    Returns True if deleted, False if not found.
    """
    col = get_unknown_persons_collection()
    contacts_col = get_unknown_contacts_collection()
    
    # Check if exists
    doc = col.find_one({"temp_id": temp_id})
    if not doc:
        return False
    
    # Delete the unknown person
    col.delete_one({"temp_id": temp_id})
    
    # Delete their contact history
    contacts_col.delete_many({"unknown_temp_id": temp_id})
    
    rprint(f"[red]Unknown person deleted:[/] {temp_id}")
    return True


def register_unknown_as_person(
    temp_id: str,
    name: str,
    role: str = "patient",
    phone: str = "",
    place: str = "",
    notes: str = "",
    additional_images: List[str] = None,  # List of base64 encoded images
    registered_by: str = "system"
) -> Dict[str, Any]:
    """Convert an unknown person to a registered person.
    
    This will:
    1. Create a new registered person with the provided name
    2. Store the face embedding for recognition
    3. Store face images (captured + additional uploaded)
    4. Transfer contact history
    5. DELETE the unknown person record
    
    Returns dict with success status and details.
    """
    from database import get_persons_collection, get_face_embeddings_collection, get_face_images_collection
    
    col = get_unknown_persons_collection()
    contacts_col = get_unknown_contacts_collection()
    persons_col = get_persons_collection()
    embeddings_col = get_face_embeddings_collection()
    images_col = get_face_images_collection()
    
    if additional_images is None:
        additional_images = []
    
    # Check if unknown person exists
    unknown_doc = col.find_one({"temp_id": temp_id})
    if not unknown_doc:
        return {"success": False, "error": f"Unknown person '{temp_id}' not found"}
    
    # Check if person name already exists
    existing = persons_col.find_one({"name": name})
    if existing:
        return {"success": False, "error": f"Person '{name}' already exists"}
    
    # Get face data from unknown person
    face_embedding = unknown_doc.get("face_embedding")
    face_snapshot = unknown_doc.get("face_snapshot_base64")
    
    # Generate auto ID based on role (P001, D001, V001, N001, W001)
    auto_person_id = generate_person_id(role)
    
    # Create the new registered person
    now = datetime.utcnow()
    person_doc = {
        "person_id": auto_person_id,  # Auto-generated ID like P001, D001, etc.
        "name": name,
        "role": role,
        "phone": phone if phone else None,
        "place": place if place else None,
        "notes": notes,
        "created_at": now,
        "registered_at": now,
        "registered_by": registered_by,
        "is_mdr": False,
        "face_trained": face_embedding is not None,
        "embedding_count": 1 if face_embedding else 0,
        "converted_from_unknown": temp_id,
    }
    person_result = persons_col.insert_one(person_doc)
    mongodb_id = str(person_result.inserted_id)
    # Store face embedding if available
    if face_embedding:
        embeddings_col.insert_one({
            "person_name": name,
            "embedding": face_embedding,
            "created_at": datetime.utcnow(),
            "source": "unknown_person_conversion",
            "original_temp_id": temp_id,
        })
    
    # Store face images - captured snapshot + any additional uploaded images
    image_count = 0
    
    # Store the captured face snapshot
    if face_snapshot:
        images_col.insert_one({
            "person_name": name,
            "image_base64": face_snapshot,
            "created_at": datetime.utcnow(),
            "source": "unknown_person_conversion",
            "original_temp_id": temp_id,
            "image_index": 0,
        })
        image_count += 1
    
    # Store additional uploaded images
    for idx, img_base64 in enumerate(additional_images):
        if img_base64:
            images_col.insert_one({
                "person_name": name,
                "image_base64": img_base64,
                "created_at": datetime.utcnow(),
                "source": "unknown_person_registration_upload",
                "image_index": image_count + idx,
            })
            image_count += 1
    
    # Update contact history to reference the new registered person
    contacts_col.update_many(
        {"unknown_temp_id": temp_id},
        {"$set": {
            "linked_to_registered": name,
            "linked_at": datetime.utcnow(),
            "original_unknown_id": temp_id,
        }}
    )
    
    # DELETE the unknown person record (data has been transferred)
    col.delete_one({"temp_id": temp_id})
    
    rprint(f"[green]Unknown person registered and removed:[/] {temp_id} → {name} ({role}) [ID: {auto_person_id}]")
    
    return {
        "success": True,
        "message": f"Successfully registered {temp_id} as {name} (ID: {auto_person_id})",
        "temp_id": temp_id,
        "person_id": auto_person_id,  # Auto-generated ID like P001, D001, etc.
        "mongodb_id": mongodb_id,  # MongoDB ObjectId
        "person_name": name,
        "role": role,
        "phone": phone,
        "place": place,
        "has_face_embedding": face_embedding is not None,
        "image_count": image_count,
    }
