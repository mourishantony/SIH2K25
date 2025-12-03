"""MDR (Multi-Drug Resistant) patient tracking and management."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Set

from config import ROOT_DIR

MDR_FILE = ROOT_DIR / "data" / "mdr_patients.json"


def _ensure_mdr_file() -> None:
    """Create MDR patients file if it doesn't exist."""
    MDR_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not MDR_FILE.exists():
        MDR_FILE.write_text(json.dumps({"mdr_patients": []}, indent=2))


def load_mdr_patients() -> Set[str]:
    """Load the set of MDR patient names."""
    _ensure_mdr_file()
    try:
        data = json.loads(MDR_FILE.read_text())
        return set(data.get("mdr_patients", []))
    except (json.JSONDecodeError, KeyError):
        return set()


def save_mdr_patients(patients: Set[str]) -> None:
    """Save the set of MDR patient names."""
    _ensure_mdr_file()
    data = {"mdr_patients": sorted(patients)}
    MDR_FILE.write_text(json.dumps(data, indent=2))


def mark_as_mdr(name: str) -> bool:
    """Mark a patient as MDR. Returns True if newly marked, False if already marked."""
    patients = load_mdr_patients()
    if name in patients:
        return False
    patients.add(name)
    save_mdr_patients(patients)
    return True


def unmark_mdr(name: str) -> bool:
    """Remove MDR status from a patient. Returns True if removed, False if not found."""
    patients = load_mdr_patients()
    if name not in patients:
        return False
    patients.remove(name)
    save_mdr_patients(patients)
    return True


def is_mdr_patient(name: str) -> bool:
    """Check if a patient is marked as MDR."""
    return name in load_mdr_patients()


def get_mdr_patients() -> list[str]:
    """Get sorted list of all MDR patients."""
    return sorted(load_mdr_patients())
