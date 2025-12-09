
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from bson import ObjectId

from routers.auth import get_current_user, require_permission
from database import get_alerts_collection

router = APIRouter()


@router.get("/")
async def get_all_alerts(
    limit: int = Query(100, ge=1, le=500),
    unread_only: bool = Query(False),
    current_user: dict = Depends(require_permission("alerts"))
):
   
    from email_alerter_mongo import get_email_alerter
    
    alerter = get_email_alerter()
    
    if unread_only:
        alerts = alerter.get_unread_alerts()
    else:
        alerts = alerter.get_all_alerts(limit=limit)
    
    return {
        "total": len(alerts),
        "alerts": alerts
    }


@router.get("/unread")
async def get_unread_alerts(current_user: dict = Depends(require_permission("alerts"))):
    
    from email_alerter_mongo import get_email_alerter
    
    alerter = get_email_alerter()
    alerts = alerter.get_unread_alerts()
    count = alerter.get_unread_count()
    
    return {
        "unread_count": count,
        "alerts": alerts
    }


@router.get("/count")
async def get_alert_counts(current_user: dict = Depends(require_permission("alerts"))):
    
    from email_alerter_mongo import get_email_alerter
    from alert_system_mongo import get_alert_system
    
    email_alerter = get_email_alerter()
    
    return {
        "mdr_alerts": {
            "total": len(email_alerter.get_all_alerts(limit=10000)),
            "unread": email_alerter.get_unread_count()
        }
    }


@router.get("/{alert_id}")
async def get_alert_detail(
    alert_id: str,
    current_user: dict = Depends(require_permission("alerts"))
):
    
    from email_alerter_mongo import get_email_alerter
    
    alerter = get_email_alerter()
    
    try:
        alert = alerter.get_alert_detail(alert_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid alert ID")
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return alert


@router.delete("/{alert_id}")
async def delete_alert(
    alert_id: str,
    current_user: dict = Depends(require_permission("alerts"))
):

    alerts = get_alerts_collection()

    try:
        oid = ObjectId(alert_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid alert ID")

    result = alerts.delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"message": "Alert deleted"}


@router.post("/{alert_id}/read")
async def mark_alert_read(
    alert_id: str,
    current_user: dict = Depends(require_permission("alerts"))
):
    
    from email_alerter_mongo import get_email_alerter
    
    alerter = get_email_alerter()
    
    try:
        success = alerter.mark_as_read(alert_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid alert ID")
    
    if success:
        return {"message": "Alert marked as read"}
    else:
        return {"message": "Alert was already read or not found"}


@router.post("/read-all")
async def mark_all_alerts_read(current_user: dict = Depends(require_permission("alerts"))):
    
    from email_alerter_mongo import get_email_alerter
    
    alerter = get_email_alerter()
    count = alerter.mark_all_as_read()
    
    return {
        "message": f"Marked {count} alerts as read",
        "count": count
    }


@router.get("/patient/{patient_name}")
async def get_alerts_for_patient(
    patient_name: str,
    current_user: dict = Depends(require_permission("alerts"))
):
    
    from email_alerter_mongo import get_email_alerter
    
    alerter = get_email_alerter()
    alerts = alerter.get_alerts_for_patient(patient_name)
    
    return {
        "patient": patient_name,
        "total": len(alerts),
        "alerts": alerts
    }


@router.get("/collision/recent")
async def get_recent_collision_alerts(
    limit: int = Query(50, ge=1, le=200),
    current_user: dict = Depends(require_permission("alerts"))
):
    
    from alert_system_mongo import get_alert_system
    
    alert_system = get_alert_system()
    alerts = alert_system.get_recent_alerts(limit=limit)
    
    return {
        "total": len(alerts),
        "alerts": alerts
    }


@router.post("/collision/{alert_id}/read")
async def mark_collision_alert_read(
    alert_id: str,
    current_user: dict = Depends(require_permission("alerts"))
):
    
    from alert_system_mongo import get_alert_system
    
    alert_system = get_alert_system()
    
    try:
        success = alert_system.mark_alert_read(alert_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid alert ID")
    
    return {"success": success}


@router.delete("/all")
async def delete_all_alerts(
    current_user: dict = Depends(get_current_user)
):
    
    
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin can delete all alerts"
        )
    
    alerts = get_alerts_collection()
    result = alerts.delete_many({})
    
    return {
        "message": f"Deleted {result.deleted_count} alerts",
        "deleted_count": result.deleted_count
    }
