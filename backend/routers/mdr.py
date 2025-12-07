"""MDR (Multi-Drug Resistant) patient management router."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from bson import ObjectId

from database import get_persons_collection, get_mdr_patients_collection
from routers.auth import get_current_user, require_permission

router = APIRouter()


class MDRMarkRequest(BaseModel):
    person_name: str
    notes: Optional[str] = None
    pathogen_type: Optional[str] = "Other"


class MDRUpdateRequest(BaseModel):
    notes: Optional[str] = None
    pathogen_type: Optional[str] = None


@router.get("/pathogens")
async def get_pathogen_types(
    current_user: dict = Depends(require_permission("mdr_management"))
):
    """Get available pathogen types and their factors."""
    from config import MDR_PATHOGEN_FACTORS
    
    pathogens = [
        {"type": k, "factor": v, "description": _get_pathogen_description(k)}
        for k, v in MDR_PATHOGEN_FACTORS.items()
    ]
    
    return {
        "pathogens": pathogens
    }


def _get_pathogen_description(pathogen_type: str) -> str:
    """Get description for pathogen type."""
    descriptions = {
        "MRSA": "Methicillin-resistant Staphylococcus aureus",
        "MDR-TB": "Multi-drug resistant Tuberculosis",
        "VRE": "Vancomycin-resistant Enterococci",
        "CRE": "Carbapenem-resistant Enterobacteriaceae",
        "ESBL": "Extended-spectrum beta-lactamase producing bacteria",
        "Other": "Other MDR pathogen",
    }
    return descriptions.get(pathogen_type, "Unknown pathogen type")


@router.get("/patients")
async def get_mdr_patients(
    current_user: dict = Depends(require_permission("mdr_management"))
):
    """Get all MDR patients."""
    from mdr_tracker_mongo import get_mdr_patients_details
    
    patients = get_mdr_patients_details()
    
    # Add additional info from persons collection
    persons = get_persons_collection()
    
    result = []
    for patient in patients:
        person = persons.find_one({"name": patient["name"]})
        result.append({
            **patient,
            "role": person.get("role") if person else "unknown",
            "phone": person.get("phone") if person else None,
            "place": person.get("place") if person else None
        })
    
    return {
        "total": len(result),
        "patients": result
    }


@router.get("/patients/{name}")
async def get_mdr_patient(
    name: str,
    current_user: dict = Depends(require_permission("mdr_management"))
):
    """Get MDR patient details with contact alerts."""
    from mdr_tracker_mongo import get_mdr_patient_info, is_mdr_patient
    from email_alerter_mongo import get_email_alerter
    
    if not is_mdr_patient(name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{name}' is not marked as MDR patient"
        )
    
    info = get_mdr_patient_info(name)
    
    # Get person details
    persons = get_persons_collection()
    person = persons.find_one({"name": name})
    
    # Get alerts for this patient
    alerter = get_email_alerter()
    alerts = alerter.get_alerts_for_patient(name)
    
    return {
        **info,
        "role": person.get("role") if person else "unknown",
        "phone": person.get("phone") if person else None,
        "place": person.get("place") if person else None,
        "alert_count": len(alerts),
        "recent_alerts": alerts[:5]
    }


@router.post("/mark")
async def mark_mdr_patient(
    data: MDRMarkRequest,
    current_user: dict = Depends(require_permission("mdr_management"))
):
    """Mark a person as MDR patient with pathogen type and backtrack contacts."""
    from mdr_tracker_mongo import mark_as_mdr, is_mdr_patient
    from config import MDR_PATHOGEN_FACTORS
    from database import get_pathogens_collection, get_alerts_collection, get_contacts_collection
    from datetime import timedelta
    
    # Check if person exists
    persons = get_persons_collection()
    person = persons.find_one({"name": data.person_name})
    
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person '{data.person_name}' not found. Please register them first."
        )
    
    # Check if already MDR
    if is_mdr_patient(data.person_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{data.person_name}' is already marked as MDR patient"
        )
    
    # Get pathogen factor from database or config
    pathogen_type = data.pathogen_type or "Other"
    pathogens_col = get_pathogens_collection()
    pathogen_doc = pathogens_col.find_one({"name": pathogen_type})
    
    if pathogen_doc:
        pathogen_factor = pathogen_doc.get("risk_factor", 1.0)
        incubation_days = pathogen_doc.get("incubation_days", 14)
    else:
        pathogen_factor = MDR_PATHOGEN_FACTORS.get(pathogen_type, 1.0)
        incubation_days = 14
    
    marked_at = datetime.now()  # Use local time for display
    
    # Mark as MDR
    success = mark_as_mdr(
        data.person_name,
        marked_by=current_user["username"],
        notes=data.notes or "",
        pathogen_type=pathogen_type,
        pathogen_factor=pathogen_factor,
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark patient as MDR"
        )
    
    # Get past contacts for backtracking
    contacts_col = get_contacts_collection()
    cutoff_date = marked_at - timedelta(days=incubation_days)
    
    query = {
        "$or": [
            {"person": data.person_name},
            {"other_person": data.person_name}
        ],
        "timestamp": {"$gte": cutoff_date}
    }
    
    contacts = list(contacts_col.find(query))
    
    # Group contacts by the other person
    past_contacts = []
    contact_summary = {}
    for contact in contacts:
        other = contact["other_person"] if contact["person"] == data.person_name else contact["person"]
        
        if other not in contact_summary:
            contact_summary[other] = {
                "person_name": other,
                "contact_count": 0,
                "total_duration": 0,
                "max_risk": 0,
                "first_contact": contact.get("timestamp"),
                "last_contact": contact.get("timestamp"),
            }
        
        summary = contact_summary[other]
        summary["contact_count"] += 1
        summary["total_duration"] += contact.get("duration_seconds", 0) or 0
        summary["max_risk"] = max(summary["max_risk"], contact.get("risk_percent", 0))
        
        if contact.get("timestamp"):
            if summary["first_contact"] is None or contact["timestamp"] < summary["first_contact"]:
                summary["first_contact"] = contact["timestamp"]
            if summary["last_contact"] is None or contact["timestamp"] > summary["last_contact"]:
                summary["last_contact"] = contact["timestamp"]
    
    # Build past contacts list with adjusted risks
    for person_name, summary in contact_summary.items():
        adjusted_risk = min(100.0, summary["max_risk"] * pathogen_factor)
        past_contacts.append({
            "person_name": person_name,
            "contact_count": summary["contact_count"],
            "total_duration": summary["total_duration"],
            "risk_percent": adjusted_risk,
            "first_contact": summary["first_contact"].isoformat() if summary["first_contact"] else None,
            "last_contact": summary["last_contact"].isoformat() if summary["last_contact"] else None,
        })
    
    # Create ONE combined alert with MDR patient info + all past contacts
    alerts_col = get_alerts_collection()
    combined_alert = {
        "mdr_patient": data.person_name,
        "alert_type": "mdr_marked",
        "pathogen_type": pathogen_type,
        "pathogen_factor": pathogen_factor,
        "incubation_days": incubation_days,
        "marked_by": current_user["username"],
        "notes": data.notes or "",
        "past_contacts": past_contacts,  # All past contacts embedded in this alert
        "past_contacts_count": len(past_contacts),
        "timestamp": marked_at,
        "created_at": marked_at,
        "read": False,
        "email_sent": False,
    }
    
    result = alerts_col.insert_one(combined_alert)
    alert_id = str(result.inserted_id)
    
    # Send ONE email with MDR patient + all past contacts
    await _send_mdr_marked_with_contacts_email(
        alert_id=alert_id,
        patient_name=data.person_name,
        pathogen_type=pathogen_type,
        pathogen_factor=pathogen_factor,
        marked_by=current_user["username"],
        marked_at=marked_at,
        past_contacts=past_contacts,
    )
    
    return {
        "message": f"'{data.person_name}' has been marked as MDR patient",
        "marked_at": marked_at.isoformat(),
        "pathogen_type": pathogen_type,
        "pathogen_factor": pathogen_factor,
        "past_contacts_found": len(past_contacts),
    }


async def _send_mdr_marked_with_contacts_email(
    alert_id: str,
    patient_name: str,
    pathogen_type: str,
    pathogen_factor: float,
    marked_by: str,
    marked_at: datetime,
    past_contacts: list,
):
    """Send email notification when a patient is marked as MDR with all past contacts."""
    import os
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from bson import ObjectId
    from database import get_alerts_collection
    
    smtp_server = os.getenv("MDR_SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("MDR_SMTP_PORT", "587"))
    smtp_username = os.getenv("MDR_SMTP_USERNAME", "")
    smtp_password = os.getenv("MDR_SMTP_PASSWORD", "")
    admin_email = os.getenv("MDR_ADMIN_EMAIL", "")
    from_email = os.getenv("MDR_FROM_EMAIL", smtp_username)
    
    if not (smtp_username and smtp_password and admin_email):
        print(f"[MDR Alert] Email skipped - SMTP not configured")
        return
    
    try:
        msg = MIMEMultipart("related")
        
        if past_contacts:
            msg["Subject"] = f"🚨 MDR Patient Alert - {patient_name} ({len(past_contacts)} past contacts found)"
        else:
            msg["Subject"] = f"🏥 New MDR Patient Marked - {patient_name}"
        
        msg["From"] = from_email
        msg["To"] = admin_email
        
        # Build past contacts HTML table
        contacts_html = ""
        if past_contacts:
            contacts_rows = ""
            for i, contact in enumerate(past_contacts, 1):
                risk = contact.get("risk_percent", 0)
                risk_color = "#c92a2a" if risk >= 40 else "#f08c00" if risk >= 20 else "#37b24d"
                duration_min = contact.get("total_duration", 0) / 60.0
                contacts_rows += f"""
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #dee2e6;">{i}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #dee2e6; font-weight: bold;">{contact.get('person_name', 'Unknown')}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #dee2e6;">{contact.get('contact_count', 1)}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #dee2e6;">{duration_min:.1f} min</td>
                    <td style="padding: 10px; border-bottom: 1px solid #dee2e6; color: {risk_color}; font-weight: bold;">{risk:.1f}%</td>
                </tr>
                """
            
            contacts_html = f"""
            <div style="margin-top: 20px;">
                <h3 style="color: #c92a2a;">⚠️ Past Contacts Found ({len(past_contacts)} people)</h3>
                <p>The following individuals had contact with this patient and may require screening:</p>
                <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                    <thead>
                        <tr style="background: #f1f3f4;">
                            <th style="padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6;">#</th>
                            <th style="padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6;">Person</th>
                            <th style="padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6;">Contacts</th>
                            <th style="padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6;">Duration</th>
                            <th style="padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6;">Risk</th>
                        </tr>
                    </thead>
                    <tbody>
                        {contacts_rows}
                    </tbody>
                </table>
            </div>
            """
        else:
            contacts_html = """
            <div style="margin-top: 20px; padding: 15px; background: #d4edda; border-radius: 5px;">
                <strong>✅ No Past Contacts Found</strong><br>
                No contacts were recorded for this patient within the incubation period.
            </div>
            """
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background: #c92a2a; color: white; padding: 20px; border-radius: 5px; }}
                .content {{ padding: 20px; background: #f8f9fa; margin-top: 20px; border-radius: 5px; }}
                .alert-box {{ background: #fff3bf; border-left: 4px solid #ffd43b; padding: 15px; margin: 15px 0; }}
                .detail-row {{ padding: 10px; border-bottom: 1px solid #dee2e6; }}
                .label {{ font-weight: bold; color: #495057; }}
                .value {{ color: #212529; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚨 MDR Patient Alert</h1>
                <p style="margin: 0;">A patient has been identified as MDR positive</p>
            </div>
            
            <div class="content">
                <div class="alert-box">
                    <strong>⚠️ IMMEDIATE ATTENTION REQUIRED</strong><br>
                    A patient has been marked as carrying a multi-drug resistant pathogen.
                    {"Past contacts have been identified and require follow-up." if past_contacts else ""}
                </div>
                
                <h3>Patient Information</h3>
                <div class="detail-row">
                    <span class="label">Patient Name:</span>
                    <span class="value" style="color: #c92a2a; font-weight: bold;">{patient_name}</span>
                </div>
                
                <div class="detail-row">
                    <span class="label">Pathogen Type:</span>
                    <span class="value" style="color: #7c3aed; font-weight: bold;">{pathogen_type}</span>
                </div>
                
                <div class="detail-row">
                    <span class="label">Risk Factor:</span>
                    <span class="value">{pathogen_factor}x</span>
                </div>
                
                <div class="detail-row">
                    <span class="label">Marked By:</span>
                    <span class="value">{marked_by}</span>
                </div>
                
                <div class="detail-row">
                    <span class="label">Marked At:</span>
                    <span class="value">{marked_at.strftime('%Y-%m-%d %H:%M:%S')} UTC</span>
                </div>
                
                {contacts_html}
                
                <div style="margin-top: 20px; padding: 15px; background: #ffe3e3; border-radius: 5px;">
                    <strong>📋 Recommended Actions:</strong><br>
                    <ul>
                        <li>Implement isolation protocols for the MDR patient</li>
                        {"<li>Contact and screen all identified past contacts</li>" if past_contacts else ""}
                        <li>Review and reinforce infection control measures</li>
                        <li>Document this case in medical records</li>
                    </ul>
                </div>
            </div>
            
            <div style="margin-top: 20px; padding: 10px; color: #868e96; font-size: 12px;">
                <p>This is an automated alert from the Patient Contact Tracing System.</p>
                <p>Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC</p>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html, "html"))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        # Update alert with email sent status
        alerts_col = get_alerts_collection()
        alerts_col.update_one(
            {"_id": ObjectId(alert_id)},
            {"$set": {"email_sent": True, "email_sent_at": datetime.utcnow()}}
        )
        
        print(f"[MDR Alert] Email sent to {admin_email} - {patient_name} with {len(past_contacts)} past contacts")
        
    except Exception as e:
        print(f"[MDR Alert] Failed to send email: {e}")


@router.post("/unmark")
async def unmark_mdr_patient(
    data: MDRMarkRequest,
    current_user: dict = Depends(require_permission("mdr_management"))
):
    """Remove MDR status from a patient."""
    from mdr_tracker_mongo import unmark_mdr, is_mdr_patient
    
    # Check if is MDR
    if not is_mdr_patient(data.person_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{data.person_name}' is not marked as MDR patient"
        )
    
    # Unmark
    success = unmark_mdr(data.person_name)
    
    if success:
        return {
            "message": f"MDR status removed from '{data.person_name}'"
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove MDR status"
        )


@router.put("/patients/{name}")
async def update_mdr_patient(
    name: str,
    data: MDRUpdateRequest,
    current_user: dict = Depends(require_permission("mdr_management"))
):
    """Update MDR patient notes."""
    from mdr_tracker_mongo import is_mdr_patient, update_mdr_notes
    
    if not is_mdr_patient(name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{name}' is not marked as MDR patient"
        )
    
    if data.notes is not None:
        update_mdr_notes(name, data.notes)
    
    return {"message": f"MDR patient '{name}' updated"}


@router.get("/check/{name}")
async def check_mdr_status(
    name: str,
    current_user: dict = Depends(require_permission("mdr_management"))
):
    """Check if a person is marked as MDR."""
    from mdr_tracker_mongo import is_mdr_patient, get_mdr_patient_info
    
    is_mdr = is_mdr_patient(name)
    info = get_mdr_patient_info(name) if is_mdr else None
    
    return {
        "name": name,
        "is_mdr": is_mdr,
        "info": info
    }


@router.get("/eligible")
async def get_eligible_for_mdr(
    current_user: dict = Depends(require_permission("mdr_management"))
):
    """Get patients who can be marked as MDR (registered but not yet MDR)."""
    persons = get_persons_collection()
    mdr = get_mdr_patients_collection()
    
    # Get all MDR patient names
    mdr_names = set(doc["name"] for doc in mdr.find({}, {"name": 1}))
    
    # Get patients not in MDR list
    eligible = []
    for doc in persons.find({"role": "patient"}).sort("name", 1):
        if doc["name"] not in mdr_names:
            eligible.append({
                "name": doc["name"],
                "phone": doc.get("phone"),
                "place": doc.get("place"),
                "face_trained": doc.get("face_trained", False)
            })
    
    return {
        "total": len(eligible),
        "eligible_patients": eligible
    }


@router.get("/contacts/{name}")
async def get_mdr_patient_contacts(
    name: str,
    limit: int = Query(50, ge=1, le=200),
    current_user: dict = Depends(require_permission("mdr_management"))
):
    """Get all contact records for an MDR patient."""
    from mdr_tracker_mongo import is_mdr_patient
    from contact_store_mongo import get_contact_ledger
    
    if not is_mdr_patient(name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{name}' is not marked as MDR patient"
        )
    
    ledger = get_contact_ledger()
    contacts = ledger.get_contacts_for_person(name)
    summary = ledger.get_contact_summary(name)
    
    return {
        "mdr_patient": name,
        "total_contacts": len(contacts),
        "contacts": contacts[:limit],
        "summary": summary
    }
