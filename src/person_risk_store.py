"""
Cumulative risk tracking between persons.

This module stores bidirectional cumulative risk scores between person pairs.
When Person A has contact with Person B:
- A's risk from B is tracked separately
- B's risk from A is tracked separately
Risk accumulates across multiple contact sessions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from database import get_person_risk_scores_collection


@dataclass
class PersonRiskData:
    """Risk data for a person due to contact with another person."""
    person: str
    other_person: str
    cumulative_risk: float  # 0.0 to 1.0 scale (multiply by 100 for percentage)
    contact_count: int
    total_duration_seconds: float
    first_contact: datetime
    last_contact: datetime
    
    @property
    def risk_percent(self) -> float:
        """Get risk as percentage (capped at 100%)."""
        return min(100.0, self.cumulative_risk * 100.0)


def _make_pair_key(person: str, other_person: str) -> Tuple[str, str]:
    """Create a sorted pair key for consistent lookup."""
    return tuple(sorted([person, other_person]))


def get_cumulative_risk(person: str, other_person: str) -> float:
    """
    Get the cumulative risk score for person due to contact with other_person.
    
    Args:
        person: The person whose risk we want to know
        other_person: The person who contributed to that risk
        
    Returns:
        The cumulative risk score (0.0 to 1.0 scale)
    """
    collection = get_person_risk_scores_collection()
    doc = collection.find_one({"person": person, "other_person": other_person})
    if doc:
        return doc.get("cumulative_risk", 0.0)
    return 0.0


def get_risk_data(person: str, other_person: str) -> Optional[PersonRiskData]:
    """
    Get full risk data for a person due to contact with another.
    
    Returns None if no contact history exists.
    """
    collection = get_person_risk_scores_collection()
    doc = collection.find_one({"person": person, "other_person": other_person})
    if doc:
        return PersonRiskData(
            person=doc["person"],
            other_person=doc["other_person"],
            cumulative_risk=doc.get("cumulative_risk", 0.0),
            contact_count=doc.get("contact_count", 0),
            total_duration_seconds=doc.get("total_duration_seconds", 0.0),
            first_contact=doc.get("first_contact", datetime.now(timezone.utc)),
            last_contact=doc.get("last_contact", datetime.now(timezone.utc)),
        )
    return None


def update_cumulative_risk(
    person: str,
    other_person: str,
    new_cumulative_risk: float,
    contact_duration_seconds: float = 0.0,
) -> float:
    """
    Update the cumulative risk score for person due to contact with other_person.
    
    Args:
        person: The person whose risk is being updated
        other_person: The person who contributed to that risk
        new_cumulative_risk: The new cumulative risk score (0.0 to 1.0)
        contact_duration_seconds: Duration of this contact session
        
    Returns:
        The updated cumulative risk score
    """
    collection = get_person_risk_scores_collection()
    now = datetime.now(timezone.utc)
    
    # Cap risk at 1.0 (100%)
    capped_risk = min(1.0, max(0.0, new_cumulative_risk))
    
    # Try to update existing document
    result = collection.update_one(
        {"person": person, "other_person": other_person},
        {
            "$set": {
                "cumulative_risk": capped_risk,
                "last_contact": now,
                "updated_at": now,
            },
            "$inc": {
                "contact_count": 1,
                "total_duration_seconds": contact_duration_seconds,
            },
            "$setOnInsert": {
                "person": person,
                "other_person": other_person,
                "first_contact": now,
                "created_at": now,
            }
        },
        upsert=True
    )
    
    return capped_risk


def get_bidirectional_risks(person_a: str, person_b: str) -> Tuple[float, float]:
    """
    Get the cumulative risks for both persons in a pair.
    
    Returns:
        Tuple of (risk_a_from_b, risk_b_from_a)
    """
    risk_a = get_cumulative_risk(person_a, person_b)
    risk_b = get_cumulative_risk(person_b, person_a)
    return risk_a, risk_b


def update_bidirectional_risks(
    person_a: str,
    person_b: str,
    risk_a_from_b: float,
    risk_b_from_a: float,
    contact_duration_seconds: float = 0.0,
) -> Tuple[float, float]:
    """
    Update cumulative risks for both persons in a pair after a contact session.
    
    Args:
        person_a: First person
        person_b: Second person
        risk_a_from_b: New cumulative risk for A due to B
        risk_b_from_a: New cumulative risk for B due to A
        contact_duration_seconds: Duration of this contact session
        
    Returns:
        Tuple of (updated_risk_a, updated_risk_b)
    """
    updated_risk_a = update_cumulative_risk(person_a, person_b, risk_a_from_b, contact_duration_seconds)
    updated_risk_b = update_cumulative_risk(person_b, person_a, risk_b_from_a, contact_duration_seconds)
    return updated_risk_a, updated_risk_b


def get_all_risks_for_person(person: str) -> Dict[str, PersonRiskData]:
    """
    Get all risk data for a person from all their contacts.
    
    Returns:
        Dict mapping other_person name to their risk data
    """
    collection = get_person_risk_scores_collection()
    result = {}
    
    for doc in collection.find({"person": person}):
        other = doc["other_person"]
        result[other] = PersonRiskData(
            person=doc["person"],
            other_person=other,
            cumulative_risk=doc.get("cumulative_risk", 0.0),
            contact_count=doc.get("contact_count", 0),
            total_duration_seconds=doc.get("total_duration_seconds", 0.0),
            first_contact=doc.get("first_contact", datetime.now(timezone.utc)),
            last_contact=doc.get("last_contact", datetime.now(timezone.utc)),
        )
    
    return result


def get_total_risk_for_person(person: str) -> float:
    """
    Get the total accumulated risk for a person from all contacts.
    
    Returns the sum of all cumulative risks (capped at 100%).
    """
    collection = get_person_risk_scores_collection()
    pipeline = [
        {"$match": {"person": person}},
        {"$group": {
            "_id": "$person",
            "total_risk": {"$sum": "$cumulative_risk"}
        }}
    ]
    
    result = list(collection.aggregate(pipeline))
    if result:
        return min(1.0, result[0].get("total_risk", 0.0))
    return 0.0


def reset_risk_between(person: str, other_person: str) -> None:
    """Reset the risk between two specific persons (for testing/admin purposes)."""
    collection = get_person_risk_scores_collection()
    collection.delete_one({"person": person, "other_person": other_person})


def reset_all_risks_for_person(person: str) -> int:
    """Reset all risks for a specific person (for testing/admin purposes)."""
    collection = get_person_risk_scores_collection()
    result = collection.delete_many({"$or": [{"person": person}, {"other_person": person}]})
    return result.deleted_count
