"""Persons CRUD router."""
from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from bson import ObjectId

from database import get_persons_collection, get_face_embeddings_collection, get_mdr_patients_collection, generate_person_id
from routers.auth import get_current_user, require_permission

router = APIRouter()


# Pydantic Models
class PersonCreate(BaseModel):
    name: str
    role: str  # patient, doctor, visitor, worker, nurse
    phone: Optional[str] = None
    place: Optional[str] = None
    notes: Optional[str] = None


class PersonUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    phone: Optional[str] = None
    place: Optional[str] = None
    notes: Optional[str] = None


class PersonResponse(BaseModel):
    id: str
    person_id: str  # Auto-generated ID like P001, D001, V001, N001, W001
    name: str
    role: str
    phone: Optional[str]
    place: Optional[str]
    notes: Optional[str]
    is_mdr: bool
    face_trained: bool
    embedding_count: int
    created_at: datetime
    updated_at: Optional[datetime]


# Helper function to convert MongoDB document to response
def person_to_response(doc: dict) -> dict:
    # Handle both created_at and registered_at (from unknown person conversion)
    created_at = doc.get("created_at") or doc.get("registered_at")
    
    return {
        "id": str(doc["_id"]),
        "person_id": doc.get("person_id", ""),  # Auto-generated ID like P001, D001, etc.
        "name": doc["name"],
        "role": doc.get("role", "patient"),
        "phone": doc.get("phone"),
        "place": doc.get("place"),
        "notes": doc.get("notes"),
        "is_mdr": doc.get("is_mdr", False),
        "face_trained": doc.get("face_trained", False),
        "embedding_count": doc.get("embedding_count", 0),
        "created_at": created_at,
        "updated_at": doc.get("updated_at"),
        "converted_from_unknown": doc.get("converted_from_unknown"),  # Track if converted from unknown
    }


@router.get("/")
async def get_persons(
    role: Optional[str] = Query(None, description="Filter by role"),
    search: Optional[str] = Query(None, description="Search by name"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    current_user: dict = Depends(get_current_user)
):
    """Get all registered persons with optional filters."""
    collection = get_persons_collection()
    
    # Build query
    query = {}
    if role:
        query["role"] = role
    if search:
        query["name"] = {"$regex": search, "$options": "i"}
    
    # Get total count
    total = collection.count_documents(query)
    
    # Get persons
    persons = []
    for doc in collection.find(query).sort("name", 1).skip(skip).limit(limit):
        persons.append(person_to_response(doc))
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "persons": persons
    }


@router.get("/roles")
async def get_roles(current_user: dict = Depends(get_current_user)):
    """Get available roles with their ID prefixes."""
    return {
        "roles": [
            {"value": "patient", "label": "Patient", "prefix": "P"},
            {"value": "doctor", "label": "Doctor", "prefix": "D"},
            {"value": "visitor", "label": "Visitor", "prefix": "V"},
            {"value": "nurse", "label": "Nurse", "prefix": "N"},
            {"value": "worker", "label": "Worker", "prefix": "W"}
        ]
    }


@router.get("/{person_id}")
async def get_person(
    person_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a single person by ID."""
    collection = get_persons_collection()
    
    try:
        doc = collection.find_one({"_id": ObjectId(person_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid person ID")
    
    if not doc:
        raise HTTPException(status_code=404, detail="Person not found")
    
    return person_to_response(doc)


@router.get("/by-name/{name}")
async def get_person_by_name(
    name: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a person by name."""
    collection = get_persons_collection()
    
    doc = collection.find_one({"name": name})
    
    if not doc:
        raise HTTPException(status_code=404, detail="Person not found")
    
    return person_to_response(doc)


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_person(
    person: PersonCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new person with auto-generated ID (P001, D001, V001, N001, W001)."""
    collection = get_persons_collection()
    
    # Check if name already exists
    if collection.find_one({"name": person.name}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Person with this name already exists"
        )
    
    # Validate role (including nurse)
    valid_roles = ["patient", "doctor", "visitor", "worker", "nurse"]
    if person.role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {valid_roles}"
        )
    
    # Generate auto ID based on role (P001, D001, V001, N001, W001)
    auto_person_id = generate_person_id(person.role)
    
    # Create person
    new_person = {
        "person_id": auto_person_id,  # Auto-generated ID like P001, D001, etc.
        "name": person.name,
        "role": person.role,
        "phone": person.phone,
        "place": person.place,
        "notes": person.notes,
        "is_mdr": False,
        "face_trained": False,
        "embedding_count": 0,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = collection.insert_one(new_person)
    new_person["_id"] = result.inserted_id
    
    return person_to_response(new_person)


@router.put("/{person_id}")
async def update_person(
    person_id: str,
    person: PersonUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update a person."""
    collection = get_persons_collection()
    
    try:
        obj_id = ObjectId(person_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid person ID")
    
    # Check if person exists
    existing = collection.find_one({"_id": obj_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Person not found")
    
    # Build update
    update_data = {"updated_at": datetime.utcnow()}
    
    if person.name is not None:
        # Check if new name conflicts with another person
        if person.name != existing["name"]:
            if collection.find_one({"name": person.name, "_id": {"$ne": obj_id}}):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Person with this name already exists"
                )
            update_data["name"] = person.name
            
            # Update related collections
            embeddings = get_face_embeddings_collection()
            embeddings.update_many(
                {"person_name": existing["name"]},
                {"$set": {"person_name": person.name}}
            )
            
            mdr = get_mdr_patients_collection()
            mdr.update_one(
                {"name": existing["name"]},
                {"$set": {"name": person.name}}
            )
    
    if person.role is not None:
        valid_roles = ["patient", "doctor", "visitor", "worker"]
        if person.role not in valid_roles:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role. Must be one of: {valid_roles}"
            )
        update_data["role"] = person.role
    
    if person.phone is not None:
        update_data["phone"] = person.phone
    
    if person.place is not None:
        update_data["place"] = person.place
    
    if person.notes is not None:
        update_data["notes"] = person.notes
    
    # Update
    collection.update_one({"_id": obj_id}, {"$set": update_data})
    
    # Get updated document
    updated = collection.find_one({"_id": obj_id})
    return person_to_response(updated)


@router.delete("/{person_id}")
async def delete_person(
    person_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a person and all related data."""
    collection = get_persons_collection()
    
    try:
        obj_id = ObjectId(person_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid person ID")
    
    # Check if person exists
    person = collection.find_one({"_id": obj_id})
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    
    person_name = person["name"]
    
    # Delete related data
    from database import get_face_images_collection
    
    embeddings = get_face_embeddings_collection()
    embeddings_deleted = embeddings.delete_many({"person_name": person_name}).deleted_count
    
    images = get_face_images_collection()
    images_deleted = images.delete_many({"person_name": person_name}).deleted_count
    
    mdr = get_mdr_patients_collection()
    mdr.delete_one({"name": person_name})
    
    # Delete person
    collection.delete_one({"_id": obj_id})
    
    return {
        "message": f"Person '{person_name}' deleted successfully",
        "embeddings_deleted": embeddings_deleted,
        "images_deleted": images_deleted
    }


@router.get("/{person_id}/contacts")
async def get_person_contacts(
    person_id: str,
    limit: int = Query(50, ge=1, le=200),
    current_user: dict = Depends(get_current_user)
):
    """Get contact history for a person."""
    collection = get_persons_collection()
    
    try:
        obj_id = ObjectId(person_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid person ID")
    
    person = collection.find_one({"_id": obj_id})
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    
    from contact_store_mongo import get_contact_ledger
    ledger = get_contact_ledger()
    
    contacts = ledger.get_contacts_for_person(person["name"])
    
    return {
        "person": person["name"],
        "total": len(contacts),
        "contacts": contacts[:limit]
    }


@router.get("/{person_id}/risk-scores")
async def get_person_risk_scores(
    person_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get cumulative risk scores for a person from all their contacts.
    
    Returns bidirectional risk data:
    - risks_from: Risk this person has accumulated FROM contact with others
    - risks_to: Risk this person has contributed TO others (not available in current data)
    """
    collection = get_persons_collection()
    
    try:
        obj_id = ObjectId(person_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid person ID")
    
    person = collection.find_one({"_id": obj_id})
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    
    person_name = person["name"]
    
    from person_risk_store import get_all_risks_for_person, get_total_risk_for_person
    
    # Get all risks FROM other people
    risks_from_others = get_all_risks_for_person(person_name)
    total_risk = get_total_risk_for_person(person_name)
    
    risks_list = []
    for other_person, risk_data in risks_from_others.items():
        risks_list.append({
            "other_person": other_person,
            "cumulative_risk": risk_data.cumulative_risk,
            "risk_percent": risk_data.risk_percent,
            "contact_count": risk_data.contact_count,
            "total_duration_seconds": risk_data.total_duration_seconds,
            "first_contact": risk_data.first_contact.isoformat() if risk_data.first_contact else None,
            "last_contact": risk_data.last_contact.isoformat() if risk_data.last_contact else None,
        })
    
    # Sort by risk percentage (highest first)
    risks_list.sort(key=lambda x: x["risk_percent"], reverse=True)
    
    return {
        "person": person_name,
        "total_risk_percent": min(100.0, total_risk * 100),
        "contact_count": len(risks_list),
        "risks": risks_list
    }


@router.get("/by-name/{name}/risk-scores")
async def get_person_risk_scores_by_name(
    name: str,
    current_user: dict = Depends(get_current_user)
):
    """Get cumulative risk scores for a person by name."""
    from person_risk_store import get_all_risks_for_person, get_total_risk_for_person
    
    # Get all risks FROM other people
    risks_from_others = get_all_risks_for_person(name)
    total_risk = get_total_risk_for_person(name)
    
    risks_list = []
    for other_person, risk_data in risks_from_others.items():
        risks_list.append({
            "other_person": other_person,
            "cumulative_risk": risk_data.cumulative_risk,
            "risk_percent": risk_data.risk_percent,
            "contact_count": risk_data.contact_count,
            "total_duration_seconds": risk_data.total_duration_seconds,
            "first_contact": risk_data.first_contact.isoformat() if risk_data.first_contact else None,
            "last_contact": risk_data.last_contact.isoformat() if risk_data.last_contact else None,
        })
    
    # Sort by risk percentage (highest first)
    risks_list.sort(key=lambda x: x["risk_percent"], reverse=True)
    
    return {
        "person": name,
        "total_risk_percent": min(100.0, total_risk * 100),
        "contact_count": len(risks_list),
        "risks": risks_list
    }
