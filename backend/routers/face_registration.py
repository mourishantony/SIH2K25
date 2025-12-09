"""Face registration router."""
from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel

from database import get_persons_collection, get_face_images_collection
from routers.auth import get_current_user, require_permission

router = APIRouter()

# Get settings from environment
FACE_REG_TOTAL_SAMPLES = int(os.getenv("FACE_REG_TOTAL_SAMPLES", "50"))


class FaceImageUpload(BaseModel):
    person_name: str
    images: List[str]  # List of base64 encoded images
    auto_train: bool = True  # Automatically start training after upload


class TrainingRequest(BaseModel):
    person_name: str
    use_gpu: Optional[bool] = None


class TrainingResponse(BaseModel):
    person_name: str
    successful: int
    failed: int
    errors: List[str]
    total_embeddings: int


@router.get("/settings")
async def get_face_settings(current_user: dict = Depends(require_permission("register_person"))):
    """Get face registration settings."""
    return {
        "total_samples": FACE_REG_TOTAL_SAMPLES,
        "min_confidence": float(os.getenv("FACE_REG_MIN_CONFIDENCE", "0.35")),
        "use_gpu": os.getenv("FACE_REG_USE_GPU", "false").lower() == "true"
    }


@router.post("/upload")
async def upload_face_images(
    data: FaceImageUpload,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(require_permission("register_person"))
):
    """Upload face images for a person and optionally trigger training."""
    persons = get_persons_collection()
    images_collection = get_face_images_collection()
    
    # Check if person exists
    person = persons.find_one({"name": data.person_name})
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person '{data.person_name}' not found. Please register the person first."
        )
    
    if not data.images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No images provided"
        )
    
    # Store images
    stored_count = 0
    for idx, img_base64 in enumerate(data.images):
        try:
            doc = {
                "person_name": data.person_name,
                "image_base64": img_base64,
                "image_index": idx,
                "trained": False,
                "created_at": datetime.utcnow()
            }
            images_collection.insert_one(doc)
            stored_count += 1
        except Exception as e:
            print(f"Error storing image {idx}: {e}")
    
    # Update person record
    total_images = images_collection.count_documents({"person_name": data.person_name})
    persons.update_one(
        {"name": data.person_name},
        {"$set": {"total_images": total_images, "updated_at": datetime.utcnow()}}
    )
    
    response = {
        "message": f"Uploaded {stored_count} images for {data.person_name}",
        "stored_count": stored_count,
        "total_images": total_images
    }
    
    # Trigger training if requested
    if data.auto_train:
        background_tasks.add_task(train_person_background, data.person_name)
        response["training_started"] = True
        response["message"] += ". Training started in background."
    
    return response


def train_person_background(person_name: str):
    """Background task to train a person's face recognition."""
    try:
        from face_trainer import train_person_from_images
        successful, failed, errors = train_person_from_images(person_name)
        print(f"[Training] {person_name}: {successful} successful, {failed} failed")
    except Exception as e:
        print(f"[Training Error] {person_name}: {e}")


@router.post("/train", response_model=TrainingResponse)
async def train_face(
    data: TrainingRequest,
    current_user: dict = Depends(require_permission("register_person"))
):
    """Manually trigger face training for a person."""
    persons = get_persons_collection()
    
    # Check if person exists
    person = persons.find_one({"name": data.person_name})
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person '{data.person_name}' not found"
        )
    
    # Run training
    from face_trainer import train_person_from_images, get_training_status
    
    successful, failed, errors = train_person_from_images(
        data.person_name,
        use_gpu=data.use_gpu
    )
    
    status_info = get_training_status(data.person_name)
    
    return TrainingResponse(
        person_name=data.person_name,
        successful=successful,
        failed=failed,
        errors=errors,
        total_embeddings=status_info["embedding_count"]
    )


@router.post("/retrain")
async def retrain_face(
    data: TrainingRequest,
    current_user: dict = Depends(require_permission("register_person"))
):
    """Retrain all images for a person (clears existing embeddings)."""
    persons = get_persons_collection()
    
    # Check if person exists
    person = persons.find_one({"name": data.person_name})
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person '{data.person_name}' not found"
        )
    
    from face_trainer import retrain_person, get_training_status
    
    successful, failed, errors = retrain_person(
        data.person_name,
        use_gpu=data.use_gpu
    )
    
    status_info = get_training_status(data.person_name)
    
    return {
        "person_name": data.person_name,
        "successful": successful,
        "failed": failed,
        "errors": errors,
        "total_embeddings": status_info["embedding_count"]
    }


@router.get("/status/{person_name}")
async def get_training_status_endpoint(
    person_name: str,
    current_user: dict = Depends(require_permission("registered_persons"))
):
    """Get training status for a person."""
    persons = get_persons_collection()
    
    # Check if person exists
    person = persons.find_one({"name": person_name})
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person '{person_name}' not found"
        )
    
    from face_trainer import get_training_status
    
    return get_training_status(person_name)


@router.delete("/images/{person_name}")
async def delete_face_images(
    person_name: str,
    current_user: dict = Depends(require_permission("register_person"))
):
    """Delete all face images and embeddings for a person."""
    persons = get_persons_collection()
    
    # Check if person exists
    person = persons.find_one({"name": person_name})
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person '{person_name}' not found"
        )
    
    from face_trainer import delete_person_training_data
    
    result = delete_person_training_data(person_name)
    
    return {
        "message": f"Deleted training data for {person_name}",
        "images_deleted": result["images_deleted"],
        "embeddings_deleted": result["embeddings_deleted"]
    }


@router.get("/registered")
async def get_registered_faces(current_user: dict = Depends(require_permission("registered_persons"))):
    """Get list of all persons with registered faces."""
    from face_db_mongo import get_all_registered_names
    
    names = get_all_registered_names()
    
    # Get details for each
    persons = get_persons_collection()
    result = []
    
    for name in names:
        person = persons.find_one({"name": name})
        if person:
            from face_trainer import get_training_status
            status = get_training_status(name)
            
            result.append({
                "name": name,
                "role": person.get("role", "unknown"),
                "embedding_count": status["embedding_count"],
                "total_images": status["total_images"],
                "trained_images": status["trained_images"]
            })
    
    return {
        "total": len(result),
        "registered_faces": result
    }
