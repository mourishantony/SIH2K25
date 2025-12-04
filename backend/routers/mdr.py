"""MDR (Multi-Drug Resistant) patient management router."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from bson import ObjectId

from database import get_persons_collection, get_mdr_patients_collection
from routers.auth import get_current_user

router = APIRouter()


class MDRMarkRequest(BaseModel):
    person_name: str
    notes: Optional[str] = None


class MDRUpdateRequest(BaseModel):
    notes: Optional[str] = None


@router.get("/patients")
async def get_mdr_patients(
    current_user: dict = Depends(get_current_user)
):
    """Get all MDR patients."""
    from mdr_tracker_mongo import get_mdr_patients_details
    
    patients = get_mdr_patients_details()
    
    # Add additional info from persons collection
    persons = get_persons_collection()
    
    result = []
    for patient in patients:
        person = persons.find_one({"name": patient["name"]})
        result.append({
            **patient,
            "role": person.get("role") if person else "unknown",
            "phone": person.get("phone") if person else None,
            "place": person.get("place") if person else None
        })
    
    return {
        "total": len(result),
        "patients": result
    }


@router.get("/patients/{name}")
async def get_mdr_patient(
    name: str,
    current_user: dict = Depends(get_current_user)
):
    """Get MDR patient details with contact alerts."""
    from mdr_tracker_mongo import get_mdr_patient_info, is_mdr_patient
    from email_alerter_mongo import get_email_alerter
    
    if not is_mdr_patient(name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{name}' is not marked as MDR patient"
        )
    
    info = get_mdr_patient_info(name)
    
    # Get person details
    persons = get_persons_collection()
    person = persons.find_one({"name": name})
    
    # Get alerts for this patient
    alerter = get_email_alerter()
    alerts = alerter.get_alerts_for_patient(name)
    
    return {
        **info,
        "role": person.get("role") if person else "unknown",
        "phone": person.get("phone") if person else None,
        "place": person.get("place") if person else None,
        "alert_count": len(alerts),
        "recent_alerts": alerts[:5]
    }


@router.post("/mark")
async def mark_mdr_patient(
    data: MDRMarkRequest,
    current_user: dict = Depends(get_current_user)
):
    """Mark a person as MDR patient."""
    from mdr_tracker_mongo import mark_as_mdr, is_mdr_patient
    
    # Check if person exists
    persons = get_persons_collection()
    person = persons.find_one({"name": data.person_name})
    
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person '{data.person_name}' not found. Please register them first."
        )
    
    # Check if already MDR
    if is_mdr_patient(data.person_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{data.person_name}' is already marked as MDR patient"
        )
    
    # Mark as MDR
    success = mark_as_mdr(
        data.person_name,
        marked_by=current_user["username"],
        notes=data.notes or ""
    )
    
    if success:
        return {
            "message": f"'{data.person_name}' has been marked as MDR patient",
            "marked_at": datetime.utcnow().isoformat()
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark patient as MDR"
        )


@router.post("/unmark")
async def unmark_mdr_patient(
    data: MDRMarkRequest,
    current_user: dict = Depends(get_current_user)
):
    """Remove MDR status from a patient."""
    from mdr_tracker_mongo import unmark_mdr, is_mdr_patient
    
    # Check if is MDR
    if not is_mdr_patient(data.person_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{data.person_name}' is not marked as MDR patient"
        )
    
    # Unmark
    success = unmark_mdr(data.person_name)
    
    if success:
        return {
            "message": f"MDR status removed from '{data.person_name}'"
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove MDR status"
        )


@router.put("/patients/{name}")
async def update_mdr_patient(
    name: str,
    data: MDRUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update MDR patient notes."""
    from mdr_tracker_mongo import is_mdr_patient, update_mdr_notes
    
    if not is_mdr_patient(name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{name}' is not marked as MDR patient"
        )
    
    if data.notes is not None:
        update_mdr_notes(name, data.notes)
    
    return {"message": f"MDR patient '{name}' updated"}


@router.get("/check/{name}")
async def check_mdr_status(
    name: str,
    current_user: dict = Depends(get_current_user)
):
    """Check if a person is marked as MDR."""
    from mdr_tracker_mongo import is_mdr_patient, get_mdr_patient_info
    
    is_mdr = is_mdr_patient(name)
    info = get_mdr_patient_info(name) if is_mdr else None
    
    return {
        "name": name,
        "is_mdr": is_mdr,
        "info": info
    }


@router.get("/eligible")
async def get_eligible_for_mdr(
    current_user: dict = Depends(get_current_user)
):
    """Get patients who can be marked as MDR (registered but not yet MDR)."""
    persons = get_persons_collection()
    mdr = get_mdr_patients_collection()
    
    # Get all MDR patient names
    mdr_names = set(doc["name"] for doc in mdr.find({}, {"name": 1}))
    
    # Get patients not in MDR list
    eligible = []
    for doc in persons.find({"role": "patient"}).sort("name", 1):
        if doc["name"] not in mdr_names:
            eligible.append({
                "name": doc["name"],
                "phone": doc.get("phone"),
                "place": doc.get("place"),
                "face_trained": doc.get("face_trained", False)
            })
    
    return {
        "total": len(eligible),
        "eligible_patients": eligible
    }


@router.get("/contacts/{name}")
async def get_mdr_patient_contacts(
    name: str,
    limit: int = Query(50, ge=1, le=200),
    current_user: dict = Depends(get_current_user)
):
    """Get all contact records for an MDR patient."""
    from mdr_tracker_mongo import is_mdr_patient
    from contact_store_mongo import get_contact_ledger
    
    if not is_mdr_patient(name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{name}' is not marked as MDR patient"
        )
    
    ledger = get_contact_ledger()
    contacts = ledger.get_contacts_for_person(name)
    summary = ledger.get_contact_summary(name)
    
    return {
        "mdr_patient": name,
        "total_contacts": len(contacts),
        "contacts": contacts[:limit],
        "summary": summary
    }
