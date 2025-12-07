"""Unknown persons router for tracking unregistered individuals."""
from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException, status, Query
from bson import ObjectId

from routers.auth import get_current_user, require_permission

router = APIRouter()


class MarkKnownRequest(BaseModel):
    person_name: str


class RegisterUnknownRequest(BaseModel):
    name: str
    role: str = "patient"
    phone: str = ""
    place: str = ""
    notes: str = ""
    additional_images: List[str] = []  # List of base64 encoded face images


@router.get("/settings")
async def get_registration_settings(
    current_user: dict = Depends(require_permission("unknown_persons"))
):
    """Get settings for unknown person registration."""
    from config import UNKNOWN_REGISTER_MAX_IMAGES
    return {
        "max_images": UNKNOWN_REGISTER_MAX_IMAGES,
    }


@router.get("/")
async def get_all_unknown_persons(
    limit: int = Query(100, ge=1, le=500),
    current_user: dict = Depends(require_permission("unknown_persons"))
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
    current_user: dict = Depends(require_permission("unknown_persons"))
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
    current_user: dict = Depends(require_permission("unknown_persons"))
):
    """Get all contacts for an unknown person."""
    from database import get_unknown_contacts_collection
    from mdr_tracker_mongo import is_mdr_patient
    
    col = get_unknown_contacts_collection()
    contacts = []
    
    for doc in col.find({"unknown_temp_id": temp_id}).sort("timestamp", -1).limit(limit):
        registered_person = doc["registered_person"]
        contacts.append({
            "id": str(doc["_id"]),
            "other_person": registered_person,
            "registered_person": registered_person,
            "other_is_mdr": is_mdr_patient(registered_person),
            "start_time": doc.get("start_time"),
            "end_time": doc.get("end_time"),
            "duration_sec": doc.get("duration_seconds", 0),
            "duration_seconds": doc.get("duration_seconds", 0),
            "risk_percent": doc.get("risk_percent", 0),
            "contact_time": doc.get("timestamp"),
            "timestamp": doc.get("timestamp"),
            "has_snapshot": "front_snapshot_base64" in doc or "side_snapshot_base64" in doc,
        })
    
    return {
        "unknown_temp_id": temp_id,
        "total": len(contacts),
        "contacts": contacts
    }


@router.post("/{temp_id}/mark-known")
async def mark_unknown_as_known_endpoint(
    temp_id: str,
    request: MarkKnownRequest,
    current_user: dict = Depends(require_permission("unknown_persons"))
):
    """Link an unknown person to a registered person."""
    from unknown_tracker_mongo import mark_unknown_as_known
    
    success = mark_unknown_as_known(temp_id, request.person_name)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to link '{temp_id}' to '{request.person_name}'. Check that both exist."
        )
    
    return {
        "success": True,
        "message": f"Successfully linked {temp_id} to {request.person_name}",
        "temp_id": temp_id,
        "linked_to": request.person_name
    }


@router.get("/by-registered/{person_name}")
async def get_unknown_contacts_for_registered_person(
    person_name: str,
    current_user: dict = Depends(require_permission("unknown_persons"))
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
    current_user: dict = Depends(require_permission("unknown_persons"))
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
    current_user: dict = Depends(require_permission("unknown_persons"))
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


@router.delete("/{temp_id}")
async def delete_unknown_person(
    temp_id: str,
    current_user: dict = Depends(require_permission("unknown_persons"))
):
    """Delete an unknown person and their contact history."""
    from unknown_tracker_mongo import delete_unknown_person as do_delete
    
    success = do_delete(temp_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown person '{temp_id}' not found"
        )
    
    return {
        "success": True,
        "message": f"Successfully deleted {temp_id}",
        "temp_id": temp_id
    }


@router.post("/{temp_id}/register")
async def register_unknown_person(
    temp_id: str,
    request: RegisterUnknownRequest,
    current_user: dict = Depends(require_permission("unknown_persons"))
):
    """Convert an unknown person to a registered person.
    
    This will:
    1. Create a new registered person with the provided name, phone, place
    2. Use the unknown person's face snapshot + any additional images for face registration
    3. Transfer contact history to the new person
    4. DELETE the unknown person record
    """
    from unknown_tracker_mongo import register_unknown_as_person
    
    result = register_unknown_as_person(
        temp_id=temp_id,
        name=request.name,
        role=request.role,
        phone=request.phone,
        place=request.place,
        notes=request.notes,
        additional_images=request.additional_images,
        registered_by=current_user.get("username", "system")
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Failed to register unknown person")
        )
    
    return result
