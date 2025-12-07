"""Pathogen management router for EHR users."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from routers.auth import require_permission
from database import get_pathogens_collection

router = APIRouter()


# Pydantic Models
class PathogenCreate(BaseModel):
    name: str
    risk_factor: float = 1.0
    incubation_days: int = 14
    description: Optional[str] = ""


class PathogenUpdate(BaseModel):
    risk_factor: Optional[float] = None
    incubation_days: Optional[int] = None
    description: Optional[str] = None


class PathogenResponse(BaseModel):
    name: str
    risk_factor: float
    incubation_days: int
    description: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@router.get("/", response_model=List[PathogenResponse])
async def list_pathogens(
    current_user: dict = Depends(require_permission("mdr_management"))
):
    """List all MDR pathogens."""
    collection = get_pathogens_collection()
    pathogens = []
    
    for doc in collection.find().sort("name", 1):
        pathogens.append({
            "name": doc["name"],
            "risk_factor": doc.get("risk_factor", 1.0),
            "incubation_days": doc.get("incubation_days", 14),
            "description": doc.get("description", ""),
            "created_at": doc.get("created_at"),
            "updated_at": doc.get("updated_at"),
        })
    
    return pathogens


@router.post("/", response_model=PathogenResponse, status_code=status.HTTP_201_CREATED)
async def create_pathogen(
    pathogen: PathogenCreate,
    current_user: dict = Depends(require_permission("pathogen_management"))
):
    """Create a new MDR pathogen type."""
    collection = get_pathogens_collection()
    
    # Check if pathogen already exists
    existing = collection.find_one({"name": pathogen.name})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Pathogen '{pathogen.name}' already exists"
        )
    
    # Create pathogen
    now = datetime.utcnow()
    doc = {
        "name": pathogen.name,
        "risk_factor": pathogen.risk_factor,
        "incubation_days": pathogen.incubation_days,
        "description": pathogen.description or "",
        "created_at": now,
        "updated_at": now,
        "created_by": current_user.get("username", "system"),
    }
    
    collection.insert_one(doc)
    
    return {
        "name": doc["name"],
        "risk_factor": doc["risk_factor"],
        "incubation_days": doc["incubation_days"],
        "description": doc["description"],
        "created_at": doc["created_at"],
        "updated_at": doc["updated_at"],
    }


@router.get("/{pathogen_name}", response_model=PathogenResponse)
async def get_pathogen(
    pathogen_name: str,
    current_user: dict = Depends(require_permission("mdr_management"))
):
    """Get a specific pathogen by name."""
    collection = get_pathogens_collection()
    
    doc = collection.find_one({"name": pathogen_name})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pathogen '{pathogen_name}' not found"
        )
    
    return {
        "name": doc["name"],
        "risk_factor": doc.get("risk_factor", 1.0),
        "incubation_days": doc.get("incubation_days", 14),
        "description": doc.get("description", ""),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


@router.put("/{pathogen_name}", response_model=PathogenResponse)
async def update_pathogen(
    pathogen_name: str,
    pathogen: PathogenUpdate,
    current_user: dict = Depends(require_permission("pathogen_management"))
):
    """Update a pathogen's risk factor or incubation period."""
    collection = get_pathogens_collection()
    
    # Check if pathogen exists
    existing = collection.find_one({"name": pathogen_name})
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pathogen '{pathogen_name}' not found"
        )
    
    # Build update document
    update_fields = {"updated_at": datetime.utcnow()}
    
    if pathogen.risk_factor is not None:
        update_fields["risk_factor"] = pathogen.risk_factor
    if pathogen.incubation_days is not None:
        update_fields["incubation_days"] = pathogen.incubation_days
    if pathogen.description is not None:
        update_fields["description"] = pathogen.description
    
    collection.update_one(
        {"name": pathogen_name},
        {"$set": update_fields}
    )
    
    # Fetch updated document
    updated = collection.find_one({"name": pathogen_name})
    
    return {
        "name": updated["name"],
        "risk_factor": updated.get("risk_factor", 1.0),
        "incubation_days": updated.get("incubation_days", 14),
        "description": updated.get("description", ""),
        "created_at": updated.get("created_at"),
        "updated_at": updated.get("updated_at"),
    }


@router.delete("/{pathogen_name}")
async def delete_pathogen(
    pathogen_name: str,
    current_user: dict = Depends(require_permission("pathogen_management"))
):
    """Delete a pathogen type."""
    collection = get_pathogens_collection()
    
    # Prevent deletion of default 'Other' pathogen
    if pathogen_name == "Other":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete the 'Other' pathogen type"
        )
    
    result = collection.delete_one({"name": pathogen_name})
    
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pathogen '{pathogen_name}' not found"
        )
    
    return {"message": f"Pathogen '{pathogen_name}' deleted successfully"}
