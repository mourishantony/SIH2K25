"""Dashboard router for statistics and overview."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from datetime import datetime, timedelta

from database import (
    get_persons_collection,
    get_mdr_patients_collection,
    get_contacts_collection,
    get_alerts_collection,
    get_face_embeddings_collection
)
from routers.auth import get_current_user

router = APIRouter()


@router.get("/stats")
async def get_dashboard_stats(current_user: dict = Depends(get_current_user)):
    """Get dashboard statistics."""
    persons = get_persons_collection()
    mdr = get_mdr_patients_collection()
    contacts = get_contacts_collection()
    alerts = get_alerts_collection()
    embeddings = get_face_embeddings_collection()
    
    # Count by role
    patients_count = persons.count_documents({"role": "patient"})
    doctors_count = persons.count_documents({"role": "doctor"})
    visitors_count = persons.count_documents({"role": "visitor"})
    nurses_count = persons.count_documents({"role": "nurse"})
    workers_count = persons.count_documents({"role": "worker"})
    total_persons = persons.count_documents({})
    
    # MDR stats
    mdr_count = mdr.count_documents({})
    
    # Contact stats
    total_contacts = contacts.count_documents({})
    high_risk_contacts = contacts.count_documents({"risk_percent": {"$gte": 40}})
    
    # Alert stats
    total_alerts = alerts.count_documents({"alert_type": "mdr_contact"})
    unread_alerts = alerts.count_documents({"alert_type": "mdr_contact", "read": False})
    
    # Registered faces
    registered_faces = len(embeddings.distinct("person_name"))
    
    # Recent activity (last 24 hours)
    yesterday = datetime.utcnow() - timedelta(hours=24)
    recent_contacts = contacts.count_documents({"timestamp": {"$gte": yesterday}})
    recent_alerts = alerts.count_documents({
        "alert_type": "mdr_contact",
        "created_at": {"$gte": yesterday}
    })
    
    return {
        "persons": {
            "total": total_persons,
            "patients": patients_count,
            "doctors": doctors_count,
            "visitors": visitors_count,
            "nurses": nurses_count,
            "workers": workers_count
        },
        "mdr": {
            "total": mdr_count
        },
        "contacts": {
            "total": total_contacts,
            "high_risk": high_risk_contacts,
            "recent_24h": recent_contacts
        },
        "alerts": {
            "total": total_alerts,
            "unread": unread_alerts,
            "recent_24h": recent_alerts
        },
        "faces": {
            "registered": registered_faces
        }
    }


@router.get("/recent-activity")
async def get_recent_activity(
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """Get recent activity for dashboard."""
    contacts = get_contacts_collection()
    alerts = get_alerts_collection()
    persons = get_persons_collection()
    
    activities = []
    
    # Recent contacts
    for doc in contacts.find().sort("timestamp", -1).limit(limit):
        activities.append({
            "type": "contact",
            "description": f"{doc['person']} had contact with {doc['other_person']}",
            "risk_percent": doc.get("risk_percent", 0),
            "timestamp": doc["timestamp"]
        })
    
    # Recent alerts
    for doc in alerts.find({"alert_type": "mdr_contact"}).sort("created_at", -1).limit(limit):
        activities.append({
            "type": "alert",
            "description": f"MDR Alert: {doc['mdr_patient']} contacted {doc['contacted_person']}",
            "risk_percent": doc.get("risk_percent", 0),
            "timestamp": doc["created_at"]
        })
    
    # Recent registrations
    for doc in persons.find().sort("created_at", -1).limit(limit):
        activities.append({
            "type": "registration",
            "description": f"New {doc['role']} registered: {doc['name']}",
            "timestamp": doc["created_at"]
        })
    
    # Sort by timestamp
    activities.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return activities[:limit]


@router.get("/mdr-summary")
async def get_mdr_summary(current_user: dict = Depends(get_current_user)):
    """Get MDR patients summary with contact info."""
    mdr = get_mdr_patients_collection()
    alerts = get_alerts_collection()
    
    summary = []
    
    for doc in mdr.find().sort("marked_at", -1):
        # Get alert count for this patient
        alert_count = alerts.count_documents({
            "alert_type": "mdr_contact",
            "mdr_patient": doc["name"]
        })
        
        # Get latest alert
        latest_alert = alerts.find_one(
            {"alert_type": "mdr_contact", "mdr_patient": doc["name"]},
            sort=[("created_at", -1)]
        )
        
        summary.append({
            "name": doc["name"],
            "marked_at": doc.get("marked_at"),
            "alert_count": alert_count,
            "latest_alert": {
                "contacted_person": latest_alert["contacted_person"],
                "created_at": latest_alert["created_at"]
            } if latest_alert else None
        })
    
    return summary


@router.get("/contact-trends")
async def get_contact_trends(
    days: int = 7,
    current_user: dict = Depends(get_current_user)
):
    """Get contact trends for the past N days."""
    contacts = get_contacts_collection()
    
    trends = []
    
    for i in range(days):
        date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=i)
        next_date = date + timedelta(days=1)
        
        count = contacts.count_documents({
            "timestamp": {"$gte": date, "$lt": next_date}
        })
        
        high_risk = contacts.count_documents({
            "timestamp": {"$gte": date, "$lt": next_date},
            "risk_percent": {"$gte": 40}
        })
        
        trends.append({
            "date": date.strftime("%Y-%m-%d"),
            "total_contacts": count,
            "high_risk_contacts": high_risk
        })
    
    trends.reverse()
    return trends
