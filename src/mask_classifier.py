from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from joblib import dump, load
from rich import print as rprint
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ROOT_DIR

_DATASET_ROOT = ROOT_DIR / "mask_datas"
_MODEL_PATH = _DATASET_ROOT / "mask_detector.joblib"
_ALLOWED_LABELS = {"with_mask": 1, "mask_weared_incorrect": 0, "without_mask": 0}
_TARGET_SIZE = (64, 64)


class _FallbackModel:
    def fit(self, *_args, **_kwargs) -> "_FallbackModel":  # pragma: no cover - deterministic behaviour
        return self

    def predict_proba(self, array: np.ndarray) -> np.ndarray:  # pragma: no cover - deterministic behaviour
        zeros = np.zeros((array.shape[0], 2), dtype=np.float32)
        zeros[:, 0] = 1.0
        return zeros


def _clamp_bbox(bbox: Tuple[int, int, int, int], shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    height, width = shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0, 0)
    return x1, y1, x2, y2


def _load_annotation(xml_path: Path) -> Iterable[Tuple[np.ndarray, int]]:
    image_path = _DATASET_ROOT / "images" / (xml_path.stem + ".png")
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    samples: List[Tuple[np.ndarray, int]] = []
    for obj in root.findall("object"):
        label_text = obj.findtext("name") or ""
        if label_text not in _ALLOWED_LABELS:
            continue
        bbox_node = obj.find("bndbox")
        if bbox_node is None:
            continue
        x1 = int(float(bbox_node.findtext("xmin", default="0")))
        y1 = int(float(bbox_node.findtext("ymin", default="0")))
        x2 = int(float(bbox_node.findtext("xmax", default="0")))
        y2 = int(float(bbox_node.findtext("ymax", default="0")))
        x1, y1, x2, y2 = _clamp_bbox((x1, y1, x2, y2), image.shape)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        samples.append((crop, _ALLOWED_LABELS[label_text]))
    return samples


def _prepare_dataset(max_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    annotations_dir = _DATASET_ROOT / "annotations"
    if not annotations_dir.exists():
        return np.empty((0, np.prod(_TARGET_SIZE) * 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    crops: List[np.ndarray] = []
    labels: List[int] = []
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        for crop, label in _load_annotation(xml_path):
            resized = cv2.resize(crop, _TARGET_SIZE, interpolation=cv2.INTER_AREA)
            crops.append(resized)
            labels.append(label)
            if len(crops) >= max_samples:
                break
        if len(crops) >= max_samples:
            break
    if not crops:
        return np.empty((0, np.prod(_TARGET_SIZE) * 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    data = np.stack(crops).astype(np.float32)
    flat = data.reshape(len(crops), -1) / 255.0
    targets = np.asarray(labels, dtype=np.float32)
    return flat, targets


def _train_model(max_samples: int) -> Pipeline:
    features, labels = _prepare_dataset(max_samples)
    if features.size == 0:
        rprint("[yellow]Mask dataset not found or empty; defaulting to fallback model.[/]")
        return _FallbackModel()
    rprint(f"[cyan]Training mask classifier on {len(features)} samples...[/]")
    pipeline: Pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=400, class_weight="balanced", solver="saga"),
            ),
        ]
    )
    pipeline.fit(features, labels)
    dump(pipeline, _MODEL_PATH)
    rprint(f"[green]Mask classifier saved to {_MODEL_PATH}[/]")
    return pipeline


class MaskClassifier:
    """Predict whether a detected face wears a mask using the Kaggle dataset."""

    def __init__(self, max_samples: int = 2500) -> None:
        self.max_samples = max_samples
        self.model = self._load_or_train()

    def _load_or_train(self):
        if _MODEL_PATH.exists():
            return load(_MODEL_PATH)
        if not _DATASET_ROOT.exists():
            rprint("[yellow]Mask dataset folder missing; using fallback probabilities.[/]")
            return _FallbackModel()
        return _train_model(self.max_samples)

    def probability(self, face_crop: np.ndarray) -> float:
        if face_crop.size == 0:
            return 0.0
        resized = cv2.resize(face_crop, _TARGET_SIZE, interpolation=cv2.INTER_AREA)
        feature = (resized.astype(np.float32).reshape(1, -1)) / 255.0
        probs = self.model.predict_proba(feature)
        if probs.ndim == 1:
            return float(probs[-1])
        return float(probs[0, -1])


__all__ = ["MaskClassifier"]
