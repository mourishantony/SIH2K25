"""Utilities for persisting face embeddings in MongoDB."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import numpy as np

from database import get_face_embeddings_collection, get_persons_collection


def load_facebank() -> Dict[str, List[np.ndarray]]:
    """Return all stored embeddings keyed by person name from MongoDB."""
    collection = get_face_embeddings_collection()
    registry: Dict[str, List[np.ndarray]] = {}
    
    for doc in collection.find():
        name = doc["person_name"]
        embedding = np.asarray(doc["embedding"], dtype=np.float32)
        registry.setdefault(name, []).append(embedding)
    
    return registry


def save_embedding(name: str, embedding: np.ndarray) -> str:
    """Save a single embedding to MongoDB. Returns the inserted ID."""
    collection = get_face_embeddings_collection()
    result = collection.insert_one({
        "person_name": name,
        "embedding": embedding.tolist(),
        "created_at": datetime.utcnow()
    })
    return str(result.inserted_id)


def upsert_embeddings(name: str, new_vectors: Iterable[np.ndarray]) -> int:
    """Append embeddings for ``name`` and return total count stored for that identity."""
    collection = get_face_embeddings_collection()
    
    # Insert all new embeddings
    docs = []
    for vector in new_vectors:
        docs.append({
            "person_name": name,
            "embedding": np.asarray(vector, dtype=np.float32).tolist(),
            "created_at": datetime.utcnow()
        })
    
    if docs:
        collection.insert_many(docs)
    
    # Return total count for this person
    return collection.count_documents({"person_name": name})


def flatten_registry(registry: Dict[str, List[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Return two arrays ``names`` and ``embeddings`` for fast inference."""
    names: List[str] = []
    vectors: List[np.ndarray] = []
    
    for name, items in registry.items():
        for vector in items:
            names.append(name)
            vectors.append(vector)
    
    if not vectors:
        return np.array([]), np.empty((0, 512), dtype=np.float32)
    
    embeddings = np.vstack(vectors).astype(np.float32)
    names_arr = np.array(names)
    return names_arr, embeddings


def remove_person(name: str) -> bool:
    """Remove all embeddings for a given person. Returns True if the person existed."""
    collection = get_face_embeddings_collection()
    result = collection.delete_many({"person_name": name})
    return result.deleted_count > 0


def get_person_embedding_count(name: str) -> int:
    """Get the number of embeddings stored for a person."""
    collection = get_face_embeddings_collection()
    return collection.count_documents({"person_name": name})


def get_all_registered_names() -> List[str]:
    """Get list of all registered person names."""
    collection = get_face_embeddings_collection()
    return collection.distinct("person_name")


def clear_all_embeddings():
    """Clear all embeddings from the database. Use with caution!"""
    collection = get_face_embeddings_collection()
    collection.delete_many({})


__all__ = [
    "load_facebank",
    "save_embedding",
    "upsert_embeddings",
    "flatten_registry",
    "remove_person",
    "get_person_embedding_count",
    "get_all_registered_names",
    "clear_all_embeddings",
]
