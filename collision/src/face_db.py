from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "face_database.json"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_facebank() -> Dict[str, List[np.ndarray]]:
    if not DB_PATH.exists():
        return {}
    with DB_PATH.open("r", encoding="utf-8") as handle:
        raw_items = json.load(handle)
    registry: Dict[str, List[np.ndarray]] = {}
    for item in raw_items:
        name = item["name"]
        embedding = np.asarray(item["embedding"], dtype=np.float32)
        registry.setdefault(name, []).append(embedding)
    return registry


def save_facebank(registry: Dict[str, Iterable[np.ndarray]]) -> None:
    _ensure_data_dir()
    payload = []
    for name, vectors in registry.items():
        for vector in vectors:
            payload.append({"name": name, "embedding": np.asarray(vector, dtype=np.float32).tolist()})
    with DB_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def upsert_embeddings(name: str, new_vectors: Iterable[np.ndarray]) -> int:
    registry = load_facebank()
    existing = registry.setdefault(name, [])
    for vector in new_vectors:
        existing.append(np.asarray(vector, dtype=np.float32))
    save_facebank(registry)
    return len(existing)


def flatten_registry(registry: Dict[str, List[np.ndarray]]):
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
