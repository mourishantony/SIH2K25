from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import typer
from rich import print as rprint

# MongoDB-based modules (use these for web integration)
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
from reid_tracker import PersonTracker, TrackInfo
from ui_utils import pick_video_file
from vision import get_analyzer
from unknown_tracker_mongo import get_unknown_tracker, UnknownPersonTracker

BBox = Tuple[int, int, int, int]
WINDOW_NAME = "Dual-View Contact Monitor"


@dataclass
class ViewSource:
    label: str
    camera_index: int
    video_path: Optional[Path]
    prompt: bool
    _resolved_path: Optional[Path] = field(default=None, init=False, repr=False)

    def resolve_path(self) -> Optional[Path]:
        if self._resolved_path is not None:
            return self._resolved_path
        path = self.video_path
        if self.prompt:
            picked = pick_video_file(f"Select {self.label} stream")
            if picked:
                path = Path(picked)
        self._resolved_path = path
        return path

    def remember(self, path: Optional[Path]) -> None:
        self._resolved_path = path


@dataclass
class FrameResult:
    frame: np.ndarray
    named_boxes: Dict[str, BBox]


@dataclass
class PairState:
    cumulative: float = 0.0
    active: bool = False
    start_iso: Optional[str] = None
    end_iso: Optional[str] = None
    # MDR tracking fields
    involves_mdr: bool = False
    mdr_patient: Optional[str] = None
    other_person: Optional[str] = None
    mdr_alert_sent: bool = False
    start_timestamp: float = 0.0
    # Unknown person tracking fields
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
    def __init__(
        self,
        *,
        source: ViewSource,
        tracker: PersonTracker,
        capture: cv2.VideoCapture,
    ) -> None:
        self.source = source
        self.tracker = tracker
        self.capture = capture
        self.track_identities: Dict[int, str] = {}
        self.identity_claims: Dict[str, int] = {}
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps > 1.0 else 30.0
        self.frame_interval = 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0
        self.source_label = self.source.label

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
    ) -> FrameResult:
        tracks = self.tracker.update(frame)
        named_boxes: Dict[str, BBox] = {}
        faces = analyzer.get(frame)
        
        # Track which track_ids have unknown faces
        unknown_track_faces: Dict[int, Tuple[np.ndarray, float, np.ndarray]] = {}  # track_id -> (crop, score, embedding)
        
        for face in faces:
            if face.det_score < min_confidence:
                continue
            x1, y1, x2, y2 = map(int, face.bbox)
            embedding = face.normed_embedding.astype(np.float32)
            identity, score = _predict_identity(embedding, names, embeddings, threshold)
            color = (0, 200, 100) if identity != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Handle unknown persons
            if identity == "Unknown":
                matched = _match_face_to_track((x1, y1, x2, y2), tracks)
                if matched is not None and unknown_tracker is not None:
                    # Store face crop for unknown person tracking
                    face_crop = _crop(frame, (x1, y1, x2, y2))
                    if face_crop.size > 0:
                        unknown_track_faces[matched.track_id] = (face_crop, face.det_score, embedding)
                    
                    # Register or update unknown person
                    state = unknown_tracker.register_unknown(
                        track_id=matched.track_id,
                        face_crop=face_crop if face_crop.size > 0 else None,
                        face_score=face.det_score,
                        face_embedding=embedding,
                        monotonic_timestamp=timestamp,
                    )
                    
                    # Use temp_id as the identity for tracking
                    temp_id = state.temp_id
                    self.track_identities[matched.track_id] = temp_id
                    
                    # Draw with temp_id
                    cv2.putText(
                        frame,
                        f"{temp_id} {score:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 100, 255),  # Orange-red for unknown
                        2,
                    )
                    
                    body_box = _face_to_body_bbox((x1, y1, x2, y2), frame.shape)
                    named_boxes[temp_id] = body_box
                else:
                    cv2.putText(
                        frame,
                        f"Unknown {score:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )
                continue
            
            cv2.putText(
                frame,
                f"{identity} {score:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            matched = _match_face_to_track((x1, y1, x2, y2), tracks)
            if matched is None:
                body_box = _face_to_body_bbox((x1, y1, x2, y2), frame.shape)
                cv2.rectangle(frame, (body_box[0], body_box[1]), (body_box[2], body_box[3]), (0, 215, 255), 2)
                cv2.putText(
                    frame,
                    identity,
                    (body_box[0], max(body_box[1] - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 215, 255),
                    2,
                )
                named_boxes[identity] = body_box
                continue
            previous_track = self.identity_claims.get(identity)
            if previous_track is not None and previous_track != matched.track_id:
                self.track_identities.pop(previous_track, None)
            self.track_identities[matched.track_id] = identity
            self.identity_claims[identity] = matched.track_id
            crop = _crop(frame, (x1, y1, x2, y2))
            mask_probability = mask_classifier.probability(crop)
            mask_memory.update(identity, mask_probability, timestamp)

        active_ids = {track.track_id for track in tracks}
        self.track_identities = {tid: name for tid, name in self.track_identities.items() if tid in active_ids}
        self.identity_claims = {name: tid for name, tid in self.identity_claims.items() if tid in active_ids}

        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            assigned = self.track_identities.get(track.track_id, f"Track {track.track_id}")
            color = (255, 191, 0) if assigned.startswith("Track") else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                assigned,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            if not assigned.startswith("Track"):
                named_boxes[assigned] = (x1, y1, x2, y2)

        cv2.putText(
            frame,
            self.source.label,
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        return FrameResult(frame=frame, named_boxes=named_boxes)


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


def _match_face_to_track(face_bbox: BBox, tracks: Sequence[TrackInfo]) -> Optional[TrackInfo]:
    if not tracks:
        return None
    best_track: Optional[TrackInfo] = None
    best_iou = 0.0
    for track in tracks:
        iou = _bbox_iou(face_bbox, track.bbox)
        if iou > best_iou:
            best_iou = iou
            best_track = track
    if best_track is not None and best_iou >= 0.05:
        return best_track
    # Fall back to center containment when IoU is too small (extreme aspect ratios).
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


def _overlap_ratio(box_a: BBox, box_b: BBox) -> float:
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
    if area_a == 0 or area_b == 0:
        return 0.0
    return inter / float(min(area_a, area_b))


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


def _combine_frames(left: np.ndarray, right: np.ndarray) -> np.ndarray:
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


def _mask_modifier(prob_a: float, prob_b: float) -> float:
    effect = contact_settings.mask_effect
    modifier_a = max(0.05, 1.0 - prob_a * effect)
    modifier_b = max(0.05, 1.0 - prob_b * effect)
    return modifier_a * modifier_b


def _update_recent_contacts(
    store: Dict[Tuple[str, str], Tuple[float, float]],
    overlaps: Dict[Tuple[str, str], float],
    timestamp: float,
    ttl: float,
) -> None:
    for pair, overlap in overlaps.items():
        store[pair] = (overlap, timestamp)
    if ttl <= 0:
        return
    stale = [pair for pair, (_, ts) in store.items() if timestamp - ts > ttl]
    for pair in stale:
        store.pop(pair, None)


def _pairs_from_collisions(collisions: Sequence[Collision], min_risk: float) -> Dict[Tuple[str, str], float]:
    pairs: Dict[Tuple[str, str], float] = {}
    for collision in collisions:
        if collision.risk_score < min_risk:
            continue
        pairs[collision.get_collision_id()] = collision.risk_score
    return pairs


def _check_mdr_alert(
    pair: Tuple[str, str],
    state: PairState,
    front_frame: np.ndarray,
    side_frame: np.ndarray,
    email_alerter: EmailAlerter,
    mdr_alert_threshold_seconds: float,
) -> None:
    """Check if MDR email alert should be sent for this contact."""
    if not state.involves_mdr or state.mdr_alert_sent:
        return
    
    # Calculate contact duration
    current_time = time.time()
    duration_seconds = current_time - state.start_timestamp
    
    # Send alert if duration exceeds threshold (5 minutes = 300 seconds by default)
    if duration_seconds >= mdr_alert_threshold_seconds:
        alert = MDRContactAlert(
            mdr_patient=state.mdr_patient or "",
            contacted_person=state.other_person or "",
            contact_start=state.start_iso or datetime.now(timezone.utc).isoformat(),
            contact_end=None,  # Still ongoing
            duration_seconds=duration_seconds,
            risk_percent=min(state.cumulative * 100, 100),
            front_snapshot=front_frame.copy(),
            side_snapshot=side_frame.copy(),
        )
        if email_alerter.send_mdr_alert(alert):
            state.mdr_alert_sent = True
            rprint(f"[bold yellow]📧 MDR alert sent:[/] {state.mdr_patient} ↔ {state.other_person}")


def _send_mdr_completion_alert(
    pair: Tuple[str, str],
    state: PairState,
    email_alerter: EmailAlerter,
    mdr_alert_threshold_seconds: float,
) -> None:
    """Send MDR alert when contact ends, if not already sent and duration >= threshold."""
    if not state.involves_mdr or state.mdr_alert_sent:
        return
    
    current_time = time.time()
    duration_seconds = current_time - state.start_timestamp
    
    # Only send if contact lasted long enough
    if duration_seconds >= mdr_alert_threshold_seconds:
        alert = MDRContactAlert(
            mdr_patient=state.mdr_patient or "",
            contacted_person=state.other_person or "",
            contact_start=state.start_iso or datetime.now(timezone.utc).isoformat(),
            contact_end=state.end_iso or datetime.now(timezone.utc).isoformat(),
            duration_seconds=duration_seconds,
            risk_percent=min(state.cumulative * 100, 100),
            front_snapshot=None,  # No frames available at contact end
            side_snapshot=None,
        )
        if email_alerter.send_mdr_alert(alert):
            state.mdr_alert_sent = True
            rprint(f"[bold yellow]📧 MDR completion alert sent:[/] {state.mdr_patient} ↔ {state.other_person}")


def monitor_contacts(
    use_gpu: bool = typer.Option(recognition_settings.use_gpu, help="Run InsightFace on GPU when available."),
    min_confidence: float = typer.Option(recognition_settings.min_confidence, help="Minimum detector confidence to accept a face."),
    threshold: float = typer.Option(recognition_settings.threshold, help="Cosine similarity threshold to accept a prediction."),
    det_size: int = typer.Option(recognition_settings.det_size, help="Square detection size fed into InsightFace."),
    base_rate: float = typer.Option(contact_settings.base_rate, help="Base risk increment per second when contact confirmed."),
    event_penalty: float = typer.Option(contact_settings.event_penalty, help="One-time penalty when a new close contact starts."),
    overlap_threshold: float = typer.Option(contact_settings.overlap_threshold, help="Minimum overlap ratio (per view) to count as contact."),
) -> None:
    registry = load_facebank()
    if not registry:
        raise typer.Exit("No embeddings found. Run register_face.py first.")
    names, embeddings = flatten_registry(registry)
    
    # Load MDR patients and initialize email alerter
    mdr_patients = load_mdr_patients()
    email_alerter = EmailAlerter()
    mdr_alert_threshold_seconds = float(os.getenv("MDR_ALERT_THRESHOLD_SECONDS", "300"))  # 5 minutes default
    
    if mdr_patients:
        rprint(f"[yellow]⚠ Monitoring {len(mdr_patients)} MDR patient(s):[/] {', '.join(sorted(mdr_patients))}")
    if email_alerter.enabled:
        rprint(f"[green]✓ MDR email alerts enabled:[/] {email_alerter.admin_email}")
    embeddings = embeddings.astype(np.float32)
    analyzer = get_analyzer((det_size, det_size), use_gpu=use_gpu)
    mask_classifier = MaskClassifier()
    ledger = ContactLedger()  # MongoDB-based, no log_dir needed
    mask_memory = MaskMemory(contact_settings.mask_decay_seconds)
    
    # Initialize unknown person tracker
    unknown_tracker = get_unknown_tracker()
    rprint(f"[cyan]✓ Unknown person tracking enabled[/]")

    front_source = ViewSource(
        label="Front",
        camera_index=dual_view_settings.front_camera_index,
        video_path=Path(dual_view_settings.front_video_path) if dual_view_settings.front_video_path else None,
        prompt=dual_view_settings.front_video_prompt,
    )
    side_source = ViewSource(
        label="Side",
        camera_index=dual_view_settings.side_camera_index,
        video_path=Path(dual_view_settings.side_video_path) if dual_view_settings.side_video_path else None,
        prompt=dual_view_settings.side_video_prompt,
    )

    front_capture, front_label = _open_capture(front_source)
    side_capture, side_label = _open_capture(side_source)
    rprint(f"[green]Front stream:[/] {front_label}")
    rprint(f"[green]Side stream:[/] {side_label}")

    front_view = ViewPipeline(source=front_source, tracker=_build_tracker(use_gpu), capture=front_capture)
    side_view = ViewPipeline(source=side_source, tracker=_build_tracker(use_gpu), capture=side_capture)

    front_collision_tracker = CollisionTracker(collision_settings.alert_duration_seconds)
    side_collision_tracker = CollisionTracker(collision_settings.alert_duration_seconds)
    alert_system = AlertSystem(
        min_risk=collision_settings.min_risk_for_alert,
        duration_threshold=collision_settings.alert_duration_seconds,
        min_alert_interval=collision_settings.alert_cooldown_seconds,
        enable_logging=True,
        enable_audio=collision_settings.enable_audio,
    )

    frame_interval = max(front_view.frame_interval, side_view.frame_interval)
    delay_ms = max(1, int(round(frame_interval * 1000)))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    pair_states: Dict[Tuple[str, str], PairState] = {}
    recent_front: Dict[Tuple[str, str], Tuple[float, float]] = {}
    recent_side: Dict[Tuple[str, str], Tuple[float, float]] = {}
    last_tick = time.monotonic()

    frame_number = 0

    try:
        while True:
            ok_front, frame_front = front_view.read()
            ok_side, frame_side = side_view.read()
            if not ok_front or not ok_side:
                rprint("[yellow]One of the streams ended; stopping monitor.[/]")
                break

            now = time.monotonic()
            elapsed = now - last_tick
            last_tick = now
            delta_t = frame_interval if frame_interval > 0 else max(elapsed, 1.0 / 30.0)

            front_result = front_view.process(
                frame_front,
                analyzer=analyzer,
                names=names,
                embeddings=embeddings,
                min_confidence=min_confidence,
                threshold=threshold,
                mask_classifier=mask_classifier,
                mask_memory=mask_memory,
                timestamp=now,
                unknown_tracker=unknown_tracker,
            )
            side_result = side_view.process(
                frame_side,
                analyzer=analyzer,
                names=names,
                embeddings=embeddings,
                min_confidence=min_confidence,
                threshold=threshold,
                mask_classifier=mask_classifier,
                mask_memory=mask_memory,
                timestamp=now,
                unknown_tracker=unknown_tracker,
            )

            front_bboxes = _to_bounding_boxes(front_result.named_boxes)
            side_bboxes = _to_bounding_boxes(side_result.named_boxes)

            front_collisions = front_collision_tracker.update_collisions(
                detect_collisions(
                    front_bboxes,
                    iou_threshold=collision_settings.iou_threshold,
                    distance_threshold=collision_settings.distance_threshold,
                    frame_width=frame_front.shape[1],
                    frame_height=frame_front.shape[0],
                )
            )
            side_collisions = side_collision_tracker.update_collisions(
                detect_collisions(
                    side_bboxes,
                    iou_threshold=collision_settings.iou_threshold,
                    distance_threshold=collision_settings.distance_threshold,
                    frame_width=frame_side.shape[1],
                    frame_height=frame_side.shape[0],
                )
            )

            front_pairs = _pairs_from_collisions(front_collisions, overlap_threshold)
            side_pairs = _pairs_from_collisions(side_collisions, overlap_threshold)
            _update_recent_contacts(recent_front, front_pairs, now, contact_settings.pair_sync_window)
            _update_recent_contacts(recent_side, side_pairs, now, contact_settings.pair_sync_window)

            verified_collision_map: Dict[Tuple[str, str], Tuple[Optional[Collision], Optional[Collision]]] = {}
            for primary, partner in verify_collision_across_cameras(front_collisions, side_collisions):
                base = primary or partner
                if base is None:
                    continue
                verified_collision_map[_pair_key(base.person1, base.person2)] = (primary, partner)

            # Determine confirmed pairs based on setting
            # If require_both_cameras is True, use intersection (both cameras must detect)
            # If False, use union (any camera detection counts)
            if contact_settings.require_both_cameras:
                confirmed_pairs = set(recent_front.keys()).intersection(recent_side.keys())
            else:
                confirmed_pairs = set(recent_front.keys()).union(recent_side.keys())

            timestamp_iso = datetime.now(timezone.utc).isoformat()
            
            # Get list of all registered person names for identifying unknown contacts
            registered_names = set(names)
            
            for pair in confirmed_pairs:
                state = pair_states.setdefault(pair, PairState())
                prob_a = mask_memory.probability(pair[0], now)
                prob_b = mask_memory.probability(pair[1], now)
                modifier = _mask_modifier(prob_a, prob_b)
                delta_risk = base_rate * modifier * delta_t
                if not state.active:
                    delta_risk += event_penalty
                    state.active = True
                    state.cumulative = 0.0
                    state.start_iso = timestamp_iso
                    state.start_timestamp = time.time()
                    
                    # Check if this pair involves an MDR patient
                    if pair[0] in mdr_patients:
                        state.involves_mdr = True
                        state.mdr_patient = pair[0]
                        state.other_person = pair[1]
                    elif pair[1] in mdr_patients:
                        state.involves_mdr = True
                        state.mdr_patient = pair[1]
                        state.other_person = pair[0]
                    
                    # Check if this pair involves an unknown person (temp ID starts with "Unknown_")
                    person_a_unknown = pair[0].startswith("Unknown_")
                    person_b_unknown = pair[1].startswith("Unknown_")
                    person_a_registered = pair[0] in registered_names
                    person_b_registered = pair[1] in registered_names
                    
                    if person_a_unknown and person_b_registered:
                        state.involves_unknown = True
                        state.unknown_temp_id = pair[0]
                        # Find track ID for this unknown person
                        for tid, name in front_view.track_identities.items():
                            if name == pair[0]:
                                state.unknown_track_id = tid
                                unknown_tracker.log_contact_start(tid, pair[1], now)
                                break
                        for tid, name in side_view.track_identities.items():
                            if name == pair[0]:
                                state.unknown_track_id = tid
                                unknown_tracker.log_contact_start(tid, pair[1], now)
                                break
                    elif person_b_unknown and person_a_registered:
                        state.involves_unknown = True
                        state.unknown_temp_id = pair[1]
                        # Find track ID for this unknown person
                        for tid, name in front_view.track_identities.items():
                            if name == pair[1]:
                                state.unknown_track_id = tid
                                unknown_tracker.log_contact_start(tid, pair[0], now)
                                break
                        for tid, name in side_view.track_identities.items():
                            if name == pair[1]:
                                state.unknown_track_id = tid
                                unknown_tracker.log_contact_start(tid, pair[0], now)
                                break
                
                state.cumulative += delta_risk
                state.end_iso = timestamp_iso
                
                # Check if MDR alert should be sent for unknown person contact
                if state.involves_unknown and state.involves_mdr and not state.unknown_alert_sent:
                    current_time = time.time()
                    duration_seconds = current_time - state.start_timestamp
                    if duration_seconds >= mdr_alert_threshold_seconds:
                        unknown_tracker.send_mdr_alert_for_unknown(
                            unknown_track_id=state.unknown_track_id,
                            mdr_patient=state.mdr_patient,
                            duration_seconds=duration_seconds,
                            risk_percent=min(state.cumulative * 100, 100),
                            front_snapshot=front_result.frame.copy(),
                            side_snapshot=side_result.frame.copy(),
                        )
                        state.unknown_alert_sent = True
                
                # Check if MDR alert should be sent (ongoing contact >= threshold)
                _check_mdr_alert(
                    pair,
                    state,
                    front_result.frame,
                    side_result.frame,
                    email_alerter,
                    mdr_alert_threshold_seconds,
                )

                _process_collision_alert(
                    pair,
                    verified_collision_map,
                    alert_system,
                    require_both=collision_settings.require_both_cameras,
                    frame_number=frame_number,
                )

            inactive_pairs = set(pair_states.keys()) - confirmed_pairs
            for pair in list(inactive_pairs):
                state = pair_states[pair]
                if state.active and state.start_iso and state.end_iso and state.cumulative > 0:
                    # Calculate contact duration
                    contact_duration = time.time() - state.start_timestamp
                    
                    # Only log contacts if duration exceeds threshold
                    if contact_duration >= mdr_alert_threshold_seconds:
                        # Send MDR completion alert if needed (contact ended)
                        _send_mdr_completion_alert(
                            pair,
                            state,
                            email_alerter,
                            mdr_alert_threshold_seconds,
                        )
                        
                        # Log unknown person contact end
                        if state.involves_unknown and state.unknown_track_id is not None:
                            registered_person = pair[0] if pair[1] == state.unknown_temp_id else pair[1]
                            unknown_tracker.log_contact_end(
                                unknown_track_id=state.unknown_track_id,
                                registered_person=registered_person,
                                cumulative_risk=state.cumulative,
                                end_monotonic_timestamp=now,
                                front_snapshot=front_result.frame.copy() if front_result else None,
                                side_snapshot=side_result.frame.copy() if side_result else None,
                            )
                        
                        for a, b in (pair, pair[::-1]):
                            ledger.log_incident(
                                a,
                                b,
                                start_time=state.start_iso,
                                end_time=state.end_iso,
                                cumulative_risk=state.cumulative,
                            )
                    else:
                        rprint(f"[dim]Contact {pair[0]} ↔ {pair[1]} duration {contact_duration:.1f}s < threshold {mdr_alert_threshold_seconds}s, not logged[/]")
                pair_states.pop(pair, None)
            
            # Periodically clean up stale unknown persons
            if frame_number % 300 == 0:  # Every ~10 seconds at 30fps
                unknown_tracker.cleanup_stale_unknowns(max_age_seconds=300.0)

            combined = _combine_frames(front_result.frame, side_result.frame)
            _draw_risk_overlay(combined, pair_states)
            cv2.imshow(WINDOW_NAME, combined)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q"):
                break
            frame_number += 1
    finally:
        _flush_active_pairs(pair_states, ledger, email_alerter, mdr_alert_threshold_seconds)
        unknown_tracker.flush_all()  # Store all unknown persons before exit
        front_capture.release()
        side_capture.release()
        cv2.destroyAllWindows()


def _pair_key(name_a: str, name_b: str) -> Tuple[str, str]:
    return tuple(sorted((name_a, name_b)))


def _to_bounding_boxes(named_boxes: Dict[str, BBox]) -> List[BoundingBox]:
    boxes: List[BoundingBox] = []
    for name, (x1, y1, x2, y2) in named_boxes.items():
        boxes.append(
            BoundingBox(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                person_name=name,
            )
        )
    return boxes


def _process_collision_alert(
    pair: Tuple[str, str],
    collision_map: Dict[Tuple[str, str], Tuple[Optional[Collision], Optional[Collision]]],
    alert_system: AlertSystem,
    *,
    require_both: bool,
    frame_number: int,
) -> None:
    entry = collision_map.get(_pair_key(*pair))
    if entry is None:
        return
    front_collision, side_collision = entry
    if require_both and not (front_collision and side_collision):
        return
    base_collision = front_collision or side_collision
    if base_collision is None:
        return
    if alert_system.should_alert(base_collision):
        alert_system.trigger_alert(
            front_collision,
            side_collision,
            frame_number=frame_number,
            verified=bool(front_collision and side_collision),
        )


def _flush_active_pairs(
    pair_states: Dict[Tuple[str, str], PairState],
    ledger: ContactLedger,
    email_alerter: Optional[EmailAlerter] = None,
    mdr_alert_threshold_seconds: float = 300.0,
) -> None:
    for pair, state in list(pair_states.items()):
        if state.active and state.start_iso and state.end_iso and state.cumulative > 0:
            # Calculate contact duration
            contact_duration = time.time() - state.start_timestamp if state.start_timestamp > 0 else 0.0
            
            # Only log contacts if duration exceeds threshold
            if contact_duration >= mdr_alert_threshold_seconds:
                # Send MDR completion alert if applicable
                if email_alerter:
                    _send_mdr_completion_alert(
                        pair,
                        state,
                        email_alerter,
                        mdr_alert_threshold_seconds,
                    )
                for a, b in (pair, pair[::-1]):
                    ledger.log_incident(
                        a,
                        b,
                        start_time=state.start_iso,
                        end_time=state.end_iso,
                        cumulative_risk=state.cumulative,
                    )
            else:
                rprint(f"[dim]Flushing contact {pair[0]} ↔ {pair[1]} duration {contact_duration:.1f}s < threshold {mdr_alert_threshold_seconds}s, not logged[/]")
        pair_states.pop(pair, None)


def _draw_risk_overlay(frame: np.ndarray, pair_states: Dict[Tuple[str, str], PairState]) -> None:
    """Draw risk overlay on the combined frame showing active contacts and risk levels."""
    entries = [(pair[0], pair[1], state.cumulative, state.active, state.involves_mdr, state.mdr_patient) 
               for pair, state in pair_states.items() if state.cumulative > 0]
    entries.sort(key=lambda item: item[2], reverse=True)
    
    y_offset = 60  # Start below the Front/Side labels
    
    for idx, (person_a, person_b, cumulative, active, involves_mdr, mdr_patient) in enumerate(entries[:6]):
        risk_percent = min(cumulative * 100, 100)
        
        # Choose colors based on MDR status and risk level
        if involves_mdr:
            # Red for MDR contacts
            color = (0, 0, 255)
            bg_color = (0, 0, 180)
            prefix = "🚨 MDR ALERT: "
        elif risk_percent >= 40:
            color = (0, 165, 255)  # Orange
            bg_color = (0, 100, 150)
            prefix = "⚠ HIGH RISK: "
        elif active:
            color = (0, 255, 255)  # Yellow
            bg_color = (0, 150, 150)
            prefix = ""
        else:
            color = (200, 200, 200)  # Gray
            bg_color = (100, 100, 100)
            prefix = ""
        
        text = f"{prefix}{person_a} ↔ {person_b}: {risk_percent:.1f}%{'*' if active else ''}"
        
        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw background rectangle
        cv2.rectangle(
            frame,
            (20, y_offset + idx * 28 - text_h - 2),
            (30 + text_w, y_offset + idx * 28 + baseline + 2),
            bg_color,
            -1
        )
        
        cv2.putText(
            frame,
            text,
            (25, y_offset + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    
    # Show "No Active Contacts" if no pairs detected
    if not entries:
        cv2.putText(
            frame,
            "No active contacts",
            (25, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
        )


def _open_capture(source: ViewSource) -> Tuple[cv2.VideoCapture, str]:
    resolved_path = source.resolve_path()
    if resolved_path is not None:
        cap = cv2.VideoCapture(str(resolved_path))
        if not cap.isOpened():
            raise typer.BadParameter(f"Cannot open video file for {source.label}: {resolved_path}")
        source.remember(resolved_path)
        return cap, f"file {resolved_path}"
    cap = cv2.VideoCapture(source.camera_index)
    if not cap.isOpened():
        raise typer.BadParameter(f"Cannot open camera index {source.camera_index} for {source.label} view")
    source.remember(None)
    return cap, f"camera {source.camera_index}"


def _build_tracker(use_gpu: bool) -> PersonTracker:
    return PersonTracker(
        model_path=str(recognition_settings.reid_model_path or "yolov8n.pt"),
        detection_confidence=recognition_settings.reid_detector_conf,
        embedder_gpu=recognition_settings.reid_embedder_gpu or use_gpu,
        nms_iou=recognition_settings.reid_nms_iou,
        box_shrink=recognition_settings.reid_box_shrink,
    )


if __name__ == "__main__":
    typer.run(monitor_contacts)
