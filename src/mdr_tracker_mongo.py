"""MDR (Multi-Drug Resistant) patient tracking and management using MongoDB."""
from __future__ import annotations

from datetime import datetime
from typing import List, Set, Optional, Dict, Any

from database import get_mdr_patients_collection, get_persons_collection


def load_mdr_patients() -> Set[str]:
    """Load the set of MDR patient names from MongoDB."""
    collection = get_mdr_patients_collection()
    return set(doc["name"] for doc in collection.find({}, {"name": 1}))


def get_mdr_patients() -> List[str]:
    """Get sorted list of all MDR patients."""
    return sorted(load_mdr_patients())


def get_mdr_patients_details() -> List[Dict[str, Any]]:
    """Get detailed list of all MDR patients with metadata."""
    collection = get_mdr_patients_collection()
    patients = []
    for doc in collection.find().sort("marked_at", -1):
        patients.append({
            "name": doc["name"],
            "marked_at": doc.get("marked_at"),
            "marked_by": doc.get("marked_by", "system"),
            "notes": doc.get("notes", "")
        })
    return patients


def mark_as_mdr(name: str, marked_by: str = "system", notes: str = "") -> bool:
    """Mark a patient as MDR. Returns True if newly marked, False if already marked."""
    collection = get_mdr_patients_collection()
    
    # Check if already marked
    existing = collection.find_one({"name": name})
    if existing:
        return False
    
    # Mark as MDR
    collection.insert_one({
        "name": name,
        "marked_at": datetime.utcnow(),
        "marked_by": marked_by,
        "notes": notes
    })
    
    # Update person's is_mdr flag
    persons = get_persons_collection()
    persons.update_one(
        {"name": name},
        {"$set": {"is_mdr": True, "mdr_marked_at": datetime.utcnow()}}
    )
    
    return True


def unmark_mdr(name: str) -> bool:
    """Remove MDR status from a patient. Returns True if removed, False if not found."""
    collection = get_mdr_patients_collection()
    
    result = collection.delete_one({"name": name})
    
    if result.deleted_count > 0:
        # Update person's is_mdr flag
        persons = get_persons_collection()
        persons.update_one(
            {"name": name},
            {"$set": {"is_mdr": False}, "$unset": {"mdr_marked_at": ""}}
        )
        return True
    
    return False


def is_mdr_patient(name: str) -> bool:
    """Check if a patient is marked as MDR."""
    collection = get_mdr_patients_collection()
    return collection.find_one({"name": name}) is not None


def get_mdr_patient_info(name: str) -> Optional[Dict[str, Any]]:
    """Get MDR patient information."""
    collection = get_mdr_patients_collection()
    doc = collection.find_one({"name": name})
    if doc:
        return {
            "name": doc["name"],
            "marked_at": doc.get("marked_at"),
            "marked_by": doc.get("marked_by", "system"),
            "notes": doc.get("notes", "")
        }
    return None


def update_mdr_notes(name: str, notes: str) -> bool:
    """Update notes for an MDR patient."""
    collection = get_mdr_patients_collection()
    result = collection.update_one(
        {"name": name},
        {"$set": {"notes": notes}}
    )
    return result.modified_count > 0


__all__ = [
    "load_mdr_patients",
    "get_mdr_patients",
    "get_mdr_patients_details",
    "mark_as_mdr",
    "unmark_mdr",
    "is_mdr_patient",
    "get_mdr_patient_info",
    "update_mdr_notes",
]
