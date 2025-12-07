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
            "notes": doc.get("notes", ""),
            "pathogen_type": doc.get("pathogen_type", "Other"),
            "pathogen_factor": doc.get("pathogen_factor", 1.0),
        })
    return patients


def mark_as_mdr(
    name: str, 
    marked_by: str = "system", 
    notes: str = "",
    pathogen_type: str = "Other",
    pathogen_factor: Optional[float] = None,
) -> bool:
    """Mark a patient as MDR. Returns True if newly marked, False if already marked."""
    from config import MDR_PATHOGEN_FACTORS
    
    collection = get_mdr_patients_collection()
    
    # Check if already marked
    existing = collection.find_one({"name": name})
    if existing:
        return False
    
    # Get pathogen factor from config if not provided
    if pathogen_factor is None:
        pathogen_factor = MDR_PATHOGEN_FACTORS.get(pathogen_type, 1.0)
    
    # Mark as MDR
    collection.insert_one({
        "name": name,
        "marked_at": datetime.now(),  # Use local time
        "marked_by": marked_by,
        "notes": notes,
        "pathogen_type": pathogen_type,
        "pathogen_factor": pathogen_factor,
    })
    
    # Update person's is_mdr flag
    persons = get_persons_collection()
    persons.update_one(
        {"name": name},
        {"$set": {
            "is_mdr": True, 
            "mdr_marked_at": datetime.now(),  # Use local time
            "pathogen_type": pathogen_type,
            "pathogen_factor": pathogen_factor,
        }}
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
    """Get MDR patient information including pathogen details."""
    collection = get_mdr_patients_collection()
    doc = collection.find_one({"name": name})
    if doc:
        return {
            "name": doc["name"],
            "marked_at": doc.get("marked_at"),
            "marked_by": doc.get("marked_by", "system"),
            "notes": doc.get("notes", ""),
            "pathogen_type": doc.get("pathogen_type", "Other"),
            "pathogen_factor": doc.get("pathogen_factor", 1.0),
        }
    return None


def get_pathogen_info(name: str) -> tuple:
    """Get pathogen type and factor for an MDR patient.
    
    Returns (pathogen_type, pathogen_factor) or ("Other", 1.0) if not found.
    """
    collection = get_mdr_patients_collection()
    doc = collection.find_one({"name": name}, {"pathogen_type": 1, "pathogen_factor": 1})
    if doc:
        return doc.get("pathogen_type", "Other"), doc.get("pathogen_factor", 1.0)
    return "Other", 1.0


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
    "get_pathogen_info",
    "update_mdr_notes",
]
