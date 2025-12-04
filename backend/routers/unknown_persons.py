"""Unknown persons router for tracking unregistered individuals."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from bson import ObjectId

from routers.auth import get_current_user

router = APIRouter()


@router.get("/")
async def get_all_unknown_persons(
    limit: int = Query(100, ge=1, le=500),
    current_user: dict = Depends(get_current_user)
):
    """Get all tracked unknown persons."""
    from unknown_tracker_mongo import get_all_unknown_persons
    
    persons = get_all_unknown_persons(limit=limit)
    
    return {
        "total": len(persons),
        "unknown_persons": persons
    }


@router.get("/{temp_id}")
async def get_unknown_person_detail(
    temp_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed info for an unknown person including face snapshot."""
    from unknown_tracker_mongo import get_unknown_person_detail
    
    person = get_unknown_person_detail(temp_id)
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown person '{temp_id}' not found"
        )
    
    return person


@router.get("/{temp_id}/contacts")
async def get_unknown_person_contacts(
    temp_id: str,
    limit: int = Query(50, ge=1, le=200),
    current_user: dict = Depends(get_current_user)
):
    """Get all contacts for an unknown person."""
    from database import get_unknown_contacts_collection
    
    col = get_unknown_contacts_collection()
    contacts = []
    
    for doc in col.find({"unknown_temp_id": temp_id}).sort("timestamp", -1).limit(limit):
        contacts.append({
            "id": str(doc["_id"]),
            "registered_person": doc["registered_person"],
            "start_time": doc.get("start_time"),
            "end_time": doc.get("end_time"),
            "duration_seconds": doc.get("duration_seconds"),
            "risk_percent": doc.get("risk_percent", 0),
            "timestamp": doc["timestamp"],
            "has_snapshot": "front_snapshot_base64" in doc or "side_snapshot_base64" in doc,
        })
    
    return {
        "unknown_temp_id": temp_id,
        "total": len(contacts),
        "contacts": contacts
    }


@router.get("/by-registered/{person_name}")
async def get_unknown_contacts_for_registered_person(
    person_name: str,
    current_user: dict = Depends(get_current_user)
):
    """Get all unknown person contacts for a registered person."""
    from unknown_tracker_mongo import get_unknown_contacts_for_person
    
    contacts = get_unknown_contacts_for_person(person_name)
    
    return {
        "registered_person": person_name,
        "total": len(contacts),
        "unknown_contacts": contacts
    }


@router.get("/mdr/{mdr_patient_name}")
async def get_unknown_contacts_with_mdr(
    mdr_patient_name: str,
    current_user: dict = Depends(get_current_user)
):
    """Get all unknown persons who had contact with an MDR patient.
    
    This is useful when a person is marked as MDR - shows all unknown
    individuals who had contact with them along with risk % and face snapshots.
    """
    from unknown_tracker_mongo import get_unknown_contacts_with_mdr_patient
    from mdr_tracker_mongo import is_mdr_patient
    
    # Verify person is MDR
    if not is_mdr_patient(mdr_patient_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{mdr_patient_name}' is not marked as MDR patient"
        )
    
    contacts = get_unknown_contacts_with_mdr_patient(mdr_patient_name)
    
    return {
        "mdr_patient": mdr_patient_name,
        "total_unknown_contacts": len(contacts),
        "unknown_persons": contacts
    }


@router.get("/contact/{contact_id}/snapshot")
async def get_contact_snapshot(
    contact_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get snapshots for a specific unknown person contact."""
    from database import get_unknown_contacts_collection
    
    col = get_unknown_contacts_collection()
    
    try:
        doc = col.find_one({"_id": ObjectId(contact_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid contact ID")
    
    if not doc:
        raise HTTPException(status_code=404, detail="Contact not found")
    
    return {
        "id": str(doc["_id"]),
        "unknown_temp_id": doc["unknown_temp_id"],
        "registered_person": doc["registered_person"],
        "front_snapshot_base64": doc.get("front_snapshot_base64"),
        "side_snapshot_base64": doc.get("side_snapshot_base64"),
    }
