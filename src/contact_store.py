from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, MutableMapping


_SANITIZE = re.compile(r"[^A-Za-z0-9_-]")


def _safe_stem(name: str) -> str:
    cleaned = _SANITIZE.sub("_", name.strip())
    return cleaned or "contact"


@dataclass
class _PersonContacts:
    person: str
    contacts: Dict[str, Dict[str, object]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, person: str, path: Path) -> "_PersonContacts":
        if not path.exists():
            return cls(person=person)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        contacts = payload.get("contacts", {})
        return cls(person=payload.get("person", person), contacts=contacts)

    def to_dict(self) -> Dict[str, object]:
        return {"person": self.person, "contacts": self.contacts}


class ContactLedger:
    """Persist bidirectional contact histories as JSON per registered person."""

    def __init__(self, root_dir: Path) -> None:
        self.root = root_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: MutableMapping[str, _PersonContacts] = {}

    def _file_for(self, person: str) -> Path:
        folder = self.root / _safe_stem(person)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / "contacts.json"

    def _get_entry(self, person: str) -> _PersonContacts:
        if person not in self._cache:
            path = self._file_for(person)
            self._cache[person] = _PersonContacts.from_file(person, path)
        return self._cache[person]

    def _flush(self, person: str) -> None:
        entry = self._cache.get(person)
        if entry is None:
            return
        path = self._file_for(person)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(entry.to_dict(), handle, indent=2)

    def log_incident(
        self,
        person: str,
        other: str,
        *,
        start_time: str,
        end_time: str,
        cumulative_risk: float,
    ) -> None:
        entry = self._get_entry(person)
        contact_record = entry.contacts.setdefault(
            other,
            {"timestamps": [], "risk_percent": 0.0},
        )
        timestamps = contact_record.setdefault("timestamps", [])
        timestamps.append(start_time)
        current_percent = float(contact_record.get("risk_percent", 0.0))
        updated_percent = current_percent + float(cumulative_risk) * 100.0
        # Cap at 100% so the number stays meaningful to operators.
        contact_record["risk_percent"] = min(100.0, updated_percent)
        self._flush(person)


__all__ = ["ContactLedger"]
