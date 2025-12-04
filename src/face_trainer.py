"""Face training service - processes images from MongoDB and creates embeddings."""
from __future__ import annotations

import base64
import time
from typing import List, Optional, Tuple
from datetime import datetime

import cv2
import numpy as np
from rich import print as rprint

from database import get_face_images_collection, get_persons_collection
from face_db_mongo import upsert_embeddings, get_person_embedding_count
from vision import get_analyzer
from config import register_settings


def decode_base64_image(base64_string: str) -> Optional[np.ndarray]:
    """Decode a base64 string to a numpy array (image)."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        rprint(f"[red]Error decoding image: {e}[/]")
        return None


def extract_face_embedding(
    image: np.ndarray,
    analyzer,
    min_confidence: float = 0.35
) -> Optional[np.ndarray]:
    """Extract face embedding from an image using InsightFace."""
    try:
        faces = analyzer.get(image)
        
        if not faces:
            return None
        
        # Select the face with highest confidence that meets threshold
        valid_faces = [f for f in faces if f.det_score >= min_confidence]
        if not valid_faces:
            return None
        
        # Get the largest face (most prominent)
        best_face = max(
            valid_faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        
        return best_face.normed_embedding.copy()
    
    except Exception as e:
        rprint(f"[red]Error extracting embedding: {e}[/]")
        return None


def train_person_from_images(
    person_name: str,
    use_gpu: bool = None
) -> Tuple[int, int, List[str]]:
    """
    Train face recognition for a person using images stored in MongoDB.
    
    Returns: (successful_count, failed_count, error_messages)
    """
    if use_gpu is None:
        use_gpu = register_settings.use_gpu
    
    min_confidence = register_settings.min_confidence
    
    rprint(f"[cyan]Starting face training for: {person_name}[/]")
    
    # Get the face analyzer
    analyzer = get_analyzer(use_gpu=use_gpu)
    
    # Get untrained images from MongoDB
    images_collection = get_face_images_collection()
    untrained_images = list(images_collection.find({
        "person_name": person_name,
        "trained": {"$ne": True}
    }))
    
    if not untrained_images:
        rprint(f"[yellow]No untrained images found for {person_name}[/]")
        return 0, 0, ["No untrained images found"]
    
    rprint(f"[cyan]Found {len(untrained_images)} untrained images[/]")
    
    successful = 0
    failed = 0
    errors = []
    embeddings = []
    trained_image_ids = []
    
    for idx, img_doc in enumerate(untrained_images):
        try:
            # Decode image
            image = decode_base64_image(img_doc["image_base64"])
            if image is None:
                failed += 1
                errors.append(f"Image {idx + 1}: Failed to decode")
                continue
            
            # Extract embedding
            embedding = extract_face_embedding(image, analyzer, min_confidence)
            if embedding is None:
                failed += 1
                errors.append(f"Image {idx + 1}: No face detected or low confidence")
                continue
            
            embeddings.append(embedding)
            trained_image_ids.append(img_doc["_id"])
            successful += 1
            
            if (idx + 1) % 10 == 0:
                rprint(f"[cyan]Processed {idx + 1}/{len(untrained_images)} images[/]")
        
        except Exception as e:
            failed += 1
            errors.append(f"Image {idx + 1}: {str(e)}")
    
    # Store embeddings in MongoDB
    if embeddings:
        total = upsert_embeddings(person_name, embeddings)
        rprint(f"[green]✓ Stored {len(embeddings)} embeddings for {person_name}. Total: {total}[/]")
        
        # Mark images as trained
        images_collection.update_many(
            {"_id": {"$in": trained_image_ids}},
            {"$set": {"trained": True, "trained_at": datetime.utcnow()}}
        )
    
    # Update person record
    persons = get_persons_collection()
    persons.update_one(
        {"name": person_name},
        {
            "$set": {
                "face_trained": True,
                "face_trained_at": datetime.utcnow(),
                "embedding_count": get_person_embedding_count(person_name)
            }
        }
    )
    
    rprint(f"[green]Training complete: {successful} successful, {failed} failed[/]")
    
    return successful, failed, errors


def store_face_image(
    person_name: str,
    image_base64: str,
    image_index: int = 0,
    has_mask: bool = False
) -> Optional[str]:
    """
    Store a face image in MongoDB for later training.
    
    Returns: document ID if successful, None otherwise
    """
    collection = get_face_images_collection()
    
    try:
        doc = {
            "person_name": person_name,
            "image_base64": image_base64,
            "image_index": image_index,
            "has_mask": has_mask,
            "trained": False,
            "created_at": datetime.utcnow()
        }
        
        result = collection.insert_one(doc)
        return str(result.inserted_id)
    
    except Exception as e:
        rprint(f"[red]Error storing image: {e}[/]")
        return None


def store_and_train_images(
    person_name: str,
    images_base64: List[str],
    use_gpu: bool = None
) -> Tuple[int, int, List[str]]:
    """
    Store images and immediately train them.
    
    Returns: (successful_count, failed_count, error_messages)
    """
    # Store all images first
    rprint(f"[cyan]Storing {len(images_base64)} images for {person_name}[/]")
    
    for idx, img_b64 in enumerate(images_base64):
        store_face_image(person_name, img_b64, image_index=idx)
    
    # Then train
    return train_person_from_images(person_name, use_gpu)


def get_training_status(person_name: str) -> dict:
    """Get training status for a person."""
    images_collection = get_face_images_collection()
    
    total_images = images_collection.count_documents({"person_name": person_name})
    trained_images = images_collection.count_documents({
        "person_name": person_name,
        "trained": True
    })
    untrained_images = total_images - trained_images
    
    embedding_count = get_person_embedding_count(person_name)
    
    return {
        "person_name": person_name,
        "total_images": total_images,
        "trained_images": trained_images,
        "untrained_images": untrained_images,
        "embedding_count": embedding_count,
        "is_trained": embedding_count > 0
    }


def retrain_person(person_name: str, use_gpu: bool = None) -> Tuple[int, int, List[str]]:
    """
    Retrain all images for a person (mark all as untrained first).
    """
    images_collection = get_face_images_collection()
    
    # Mark all images as untrained
    images_collection.update_many(
        {"person_name": person_name},
        {"$set": {"trained": False}}
    )
    
    # Clear existing embeddings
    from face_db_mongo import remove_person
    remove_person(person_name)
    
    # Retrain
    return train_person_from_images(person_name, use_gpu)


def delete_person_training_data(person_name: str) -> dict:
    """Delete all training data for a person."""
    from face_db_mongo import remove_person
    
    images_collection = get_face_images_collection()
    
    # Delete images
    images_result = images_collection.delete_many({"person_name": person_name})
    
    # Delete embeddings
    embeddings_deleted = remove_person(person_name)
    
    # Update person record
    persons = get_persons_collection()
    persons.update_one(
        {"name": person_name},
        {
            "$set": {
                "face_trained": False,
                "embedding_count": 0
            },
            "$unset": {"face_trained_at": ""}
        }
    )
    
    return {
        "images_deleted": images_result.deleted_count,
        "embeddings_deleted": embeddings_deleted
    }


__all__ = [
    "decode_base64_image",
    "extract_face_embedding",
    "train_person_from_images",
    "store_face_image",
    "store_and_train_images",
    "get_training_status",
    "retrain_person",
    "delete_person_training_data",
]
