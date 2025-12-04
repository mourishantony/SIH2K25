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
)


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
    ) -> UnknownPersonState:
        """Register or update an unknown person.
        
        Args:
            track_id: Track ID from object tracker
            face_crop: Face image crop
            face_score: Face detection confidence score
            face_embedding: Face embedding vector
            monotonic_timestamp: time.monotonic() value for duration calculations
        """
        unix_now = time.time()  # Always use current Unix time for storage
        
        # Check if this track_id already exists
        if track_id in self._active_unknowns:
            state = self._active_unknowns[track_id]
            state.last_seen = unix_now
            state.last_seen_monotonic = monotonic_timestamp
            
            # Update face snapshot if better quality
            if face_crop is not None and face_score > state.best_face_score:
                state.best_face_snapshot = face_crop.copy()
                state.best_face_score = face_score
                if face_embedding is not None:
                    state.face_embedding = face_embedding.copy()
            
            return state
        
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
        
        # Create new unknown person
        temp_id = self._generate_temp_id()
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
        
        rprint(f"[cyan]New unknown person detected:[/] {temp_id}")
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
    
    # Create the new registered person
    now = datetime.utcnow()
    person_doc = {
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
    person_id = str(person_result.inserted_id)
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
    
    rprint(f"[green]Unknown person registered and removed:[/] {temp_id} → {name} ({role})")
    
    return {
        "success": True,
        "message": f"Successfully registered {temp_id} as {name}",
        "temp_id": temp_id,
        "person_id": person_id,
        "person_name": name,
        "role": role,
        "phone": phone,
        "place": place,
        "has_face_embedding": face_embedding is not None,
        "image_count": image_count,
    }
