"""Contact tracing storage using MongoDB."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Any, Optional

from database import get_contacts_collection


class ContactLedgerMongo:
    """Persist bidirectional contact histories in MongoDB."""

    def __init__(self):
        self.collection = get_contacts_collection()

    def log_incident(
        self,
        person: str,
        other: str,
        *,
        start_time: str,
        end_time: str,
        cumulative_risk: float,
        mdr_risk_score: float = 0.0,
        pathogen_type: str = None,
        pathogen_factor: float = None,
        is_mdr_contact: bool = False,
        distance_meters: float = None,  # Real-world distance in meters
        min_distance_meters: float = None,  # Minimum distance during contact
    ) -> str:
        """Log a contact incident. Returns the inserted document ID."""
        doc = {
            "person": person,
            "other_person": other,
            "start_time": start_time,
            "end_time": end_time,
            "cumulative_risk": cumulative_risk,
            "risk_percent": min(100.0, cumulative_risk * 100.0),
            "timestamp": datetime.utcnow(),
            "created_at": datetime.utcnow(),
            # MDR specific fields
            "is_mdr_contact": is_mdr_contact,
            "mdr_risk_score": mdr_risk_score,
            "pathogen_type": pathogen_type,
            "pathogen_factor": pathogen_factor,
            # Distance fields
            "distance_meters": distance_meters,
            "min_distance_meters": min_distance_meters,
        }
        result = self.collection.insert_one(doc)
        return str(result.inserted_id)

    def get_contacts_for_person(self, person: str) -> List[Dict[str, Any]]:
        """Get all contacts for a specific person, aggregated by contact person."""
        # Aggregate contacts by other_person
        pipeline = [
            {"$match": {"person": person}},
            {"$group": {
                "_id": "$other_person",
                "contact_count": {"$sum": 1},
                "total_risk": {"$sum": "$risk_percent"},
                "max_risk": {"$max": "$risk_percent"},
                "first_contact": {"$min": "$timestamp"},
                "last_contact": {"$max": "$timestamp"},
                "total_duration_seconds": {"$sum": {
                    "$divide": [
                        {"$subtract": [
                            {"$dateFromString": {"dateString": "$end_time"}},
                            {"$dateFromString": {"dateString": "$start_time"}}
                        ]},
                        1000
                    ]
                }}
            }},
            {"$sort": {"last_contact": -1}}
        ]
        
        contacts = []
        for doc in self.collection.aggregate(pipeline):
            duration = doc.get("total_duration_seconds", 0)
            if duration and isinstance(duration, (int, float)):
                duration = round(duration, 1)
            else:
                duration = None
            contacts.append({
                "contact_name": doc["_id"],
                "contact_count": doc["contact_count"],
                "max_risk_percent": round(doc.get("max_risk", 0), 1),
                "avg_risk_percent": round(doc.get("total_risk", 0) / doc["contact_count"], 1) if doc["contact_count"] > 0 else 0,
                "duration_seconds": duration,
                "first_contact": doc["first_contact"],
                "last_contact": doc["last_contact"],
                "timestamp": doc["last_contact"]
            })
        return contacts

    def get_contacts_between(self, person1: str, person2: str) -> List[Dict[str, Any]]:
        """Get all contacts between two specific persons."""
        contacts = []
        query = {
            "$or": [
                {"person": person1, "other_person": person2},
                {"person": person2, "other_person": person1}
            ]
        }
        for doc in self.collection.find(query).sort("timestamp", -1):
            contacts.append({
                "id": str(doc["_id"]),
                "person": doc["person"],
                "other_person": doc["other_person"],
                "start_time": doc["start_time"],
                "end_time": doc["end_time"],
                "risk_percent": doc.get("risk_percent", 0),
                "timestamp": doc["timestamp"]
            })
        return contacts

    def get_all_contacts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all contact records with optional limit."""
        contacts = []
        for doc in self.collection.find().sort("timestamp", -1).limit(limit):
            contacts.append({
                "id": str(doc["_id"]),
                "person": doc["person"],
                "other_person": doc["other_person"],
                "start_time": doc["start_time"],
                "end_time": doc["end_time"],
                "risk_percent": doc.get("risk_percent", 0),
                "timestamp": doc["timestamp"]
            })
        return contacts

    def get_contact_summary(self, person: str) -> Dict[str, Any]:
        """Get contact summary for a person."""
        pipeline = [
            {"$match": {"person": person}},
            {"$group": {
                "_id": "$other_person",
                "total_contacts": {"$sum": 1},
                "total_risk": {"$sum": "$risk_percent"},
                "last_contact": {"$max": "$timestamp"}
            }},
            {"$sort": {"total_risk": -1}}
        ]
        
        summary = {}
        for doc in self.collection.aggregate(pipeline):
            summary[doc["_id"]] = {
                "total_contacts": doc["total_contacts"],
                "total_risk_percent": min(100.0, doc["total_risk"]),
                "last_contact": doc["last_contact"]
            }
        return summary

    def get_high_risk_contacts(self, min_risk_percent: float = 40.0) -> List[Dict[str, Any]]:
        """Get contacts with risk above threshold."""
        contacts = []
        for doc in self.collection.find(
            {"risk_percent": {"$gte": min_risk_percent}}
        ).sort("risk_percent", -1):
            contacts.append({
                "id": str(doc["_id"]),
                "person": doc["person"],
                "other_person": doc["other_person"],
                "start_time": doc["start_time"],
                "end_time": doc["end_time"],
                "risk_percent": doc["risk_percent"],
                "timestamp": doc["timestamp"]
            })
        return contacts

    def get_contacts_count(self) -> int:
        """Get total number of contact records."""
        return self.collection.count_documents({})

    def clear_contacts_for_person(self, person: str) -> int:
        """Delete all contacts for a person. Returns deleted count."""
        result = self.collection.delete_many({
            "$or": [
                {"person": person},
                {"other_person": person}
            ]
        })
        return result.deleted_count


# Create a singleton instance
_ledger_instance: Optional[ContactLedgerMongo] = None


def get_contact_ledger() -> ContactLedgerMongo:
    """Get or create the contact ledger instance."""
    global _ledger_instance
    if _ledger_instance is None:
        _ledger_instance = ContactLedgerMongo()
    return _ledger_instance


__all__ = ["ContactLedgerMongo", "get_contact_ledger"]
