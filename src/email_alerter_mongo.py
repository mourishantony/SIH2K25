"""Email alerting system with MongoDB storage for MDR patient contact notifications."""
from __future__ import annotations

import os
import smtplib
import base64
from dataclasses import dataclass
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
from dotenv import load_dotenv
from rich import print as rprint

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)

from database import get_alerts_collection


@dataclass
class MDRContactAlert:
    """Data structure for MDR contact alerts."""
    mdr_patient: str
    contacted_person: str
    contact_start: str  # ISO timestamp
    contact_end: Optional[str]  # ISO timestamp or None if ongoing
    duration_seconds: float
    risk_percent: float
    front_snapshot: Optional[np.ndarray] = None
    side_snapshot: Optional[np.ndarray] = None
    distance_meters: Optional[float] = None  # Real-world distance in meters
    min_distance_meters: Optional[float] = None  # Minimum distance during contact


class EmailAlerterMongo:
    """Send email alerts for MDR patient contacts and store in MongoDB."""
    
    def __init__(self):
        self.smtp_server = os.getenv("MDR_SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("MDR_SMTP_PORT", "587"))
        self.smtp_username = os.getenv("MDR_SMTP_USERNAME", "")
        self.smtp_password = os.getenv("MDR_SMTP_PASSWORD", "")
        self.admin_email = os.getenv("MDR_ADMIN_EMAIL", "")
        self.from_email = os.getenv("MDR_FROM_EMAIL", self.smtp_username)
        self.enabled = bool(self.smtp_username and self.smtp_password and self.admin_email)
        self.collection = get_alerts_collection()
        
        if not self.enabled:
            rprint("[yellow]⚠ MDR email alerts disabled: missing SMTP credentials in .env[/]")
    
    def send_mdr_alert(self, alert: MDRContactAlert) -> Optional[str]:
        """Send MDR contact alert email and store in MongoDB. Returns document ID."""
        # Store in MongoDB first
        doc_id = self._store_alert(alert)
        
        if not self.enabled:
            rprint("[yellow]MDR email alert skipped (not configured)[/]")
            return doc_id
        
        try:
            msg = self._create_email(alert)
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            # Update MongoDB with email sent status
            from bson import ObjectId
            self.collection.update_one(
                {"_id": ObjectId(doc_id)},
                {"$set": {"email_sent": True, "email_sent_at": datetime.utcnow()}}
            )
            
            rprint(f"[bold green]✓[/] MDR alert email sent to {self.admin_email}")
            return doc_id
            
        except Exception as e:
            rprint(f"[bold red]✗[/] Failed to send MDR alert email: {e}")
            # Update MongoDB with email failure
            from bson import ObjectId
            self.collection.update_one(
                {"_id": ObjectId(doc_id)},
                {"$set": {"email_sent": False, "email_error": str(e)}}
            )
            return doc_id
    
    def _store_alert(self, alert: MDRContactAlert) -> str:
        """Store alert in MongoDB. Returns document ID."""
        doc = {
            "mdr_patient": alert.mdr_patient,
            "contacted_person": alert.contacted_person,
            "contact_start": alert.contact_start,
            "contact_end": alert.contact_end,
            "duration_seconds": alert.duration_seconds,
            "risk_percent": alert.risk_percent,
            "distance_meters": alert.distance_meters,
            "min_distance_meters": alert.min_distance_meters,
            "status": "ongoing" if alert.contact_end is None else "completed",
            "created_at": datetime.now(),  # Use local time
            "alert_created_at": datetime.now(),  # When the alert was triggered (local time)
            "read": False,
            "email_sent": False,
            "alert_type": "mdr_contact"
        }
        
        # Store snapshots as base64 if available
        if alert.front_snapshot is not None:
            success, buffer = cv2.imencode('.jpg', alert.front_snapshot)
            if success:
                doc["front_snapshot_base64"] = base64.b64encode(buffer).decode('utf-8')
        
        if alert.side_snapshot is not None:
            success, buffer = cv2.imencode('.jpg', alert.side_snapshot)
            if success:
                doc["side_snapshot_base64"] = base64.b64encode(buffer).decode('utf-8')
        
        result = self.collection.insert_one(doc)
        return str(result.inserted_id)
    
    def _create_email(self, alert: MDRContactAlert) -> MIMEMultipart:
        """Create the email message with alert details and snapshots."""
        msg = MIMEMultipart("related")
        msg["Subject"] = f"🚨 MDR Patient Contact Alert - {alert.mdr_patient}"
        msg["From"] = self.from_email
        msg["To"] = self.admin_email
        
        html_body = self._create_html_body(alert)
        msg.attach(MIMEText(html_body, "html"))
        
        if alert.front_snapshot is not None:
            self._attach_snapshot(msg, alert.front_snapshot, "front_view", "Front View")
        if alert.side_snapshot is not None:
            self._attach_snapshot(msg, alert.side_snapshot, "side_view", "Side View")
        
        return msg
    
    def _create_html_body(self, alert: MDRContactAlert) -> str:
        """Create HTML email body with alert details."""
        duration_min = alert.duration_seconds / 60.0
        status = "ONGOING" if alert.contact_end is None else "COMPLETED"
        status_color = "#ff6b6b" if status == "ONGOING" else "#51cf66"
        
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
                .risk-high {{ color: #c92a2a; font-weight: bold; }}
                .risk-medium {{ color: #f08c00; font-weight: bold; }}
                .risk-low {{ color: #37b24d; font-weight: bold; }}
                .status {{ display: inline-block; padding: 5px 10px; background: {status_color}; color: white; border-radius: 3px; }}
                .snapshot {{ margin: 10px 0; text-align: center; }}
                .snapshot img {{ max-width: 100%; border: 2px solid #dee2e6; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚨 MDR Patient Contact Alert</h1>
                <p style="margin: 0;">Multi-Drug Resistant Pathogen Exposure Warning</p>
            </div>
            
            <div class="content">
                <div class="alert-box">
                    <strong>⚠️ IMMEDIATE ATTENTION REQUIRED</strong><br>
                    An MDR-positive patient has been in close contact with another individual.
                </div>
                
                <div class="detail-row">
                    <span class="label">MDR Patient:</span>
                    <span class="value" style="color: #c92a2a; font-weight: bold;">{alert.mdr_patient}</span>
                </div>
                
                <div class="detail-row">
                    <span class="label">Contacted Person:</span>
                    <span class="value">{alert.contacted_person}</span>
                </div>
                
                <div class="detail-row">
                    <span class="label">Contact Status:</span>
                    <span class="status">{status}</span>
                </div>
                
                <div class="detail-row">
                    <span class="label">Contact Started:</span>
                    <span class="value">{alert.contact_start}</span>
                </div>
                
                <div class="detail-row">
                    <span class="label">Contact Ended:</span>
                    <span class="value">{alert.contact_end if alert.contact_end else 'Still in contact'}</span>
                </div>
                
                <div class="detail-row">
                    <span class="label">Duration:</span>
                    <span class="value">{duration_min:.1f} minutes ({alert.duration_seconds:.0f} seconds)</span>
                </div>
                
                <div class="detail-row">
                    <span class="label">Contact Distance:</span>
                    <span class="value">{f"{alert.min_distance_meters:.2f} meters (minimum)" if alert.min_distance_meters is not None else (f"{alert.distance_meters:.2f} meters" if alert.distance_meters is not None else "Not available")}</span>
                </div>
                
                <div class="detail-row">
                    <span class="label">Risk Percentage:</span>
                    <span class="{self._get_risk_class(alert.risk_percent)}">{alert.risk_percent:.1f}%</span>
                </div>
                
                <div style="margin-top: 20px; padding: 15px; background: #e7f5ff; border-radius: 5px;">
                    <h3 style="margin-top: 0;">Recommended Actions:</h3>
                    <ul>
                        <li>Isolate and assess the contacted individual: <strong>{alert.contacted_person}</strong></li>
                        <li>Conduct screening for MDR pathogens</li>
                        <li>Implement enhanced PPE protocols</li>
                        <li>Review and reinforce infection control measures</li>
                        <li>Document this exposure in the patient's medical record</li>
                    </ul>
                </div>
                
                {"<div class='snapshot'><h3>Camera Views at Time of Contact</h3>" if alert.front_snapshot is not None or alert.side_snapshot is not None else ""}
                {f"<p><strong>Front View:</strong></p><img src='cid:front_view' alt='Front View'>" if alert.front_snapshot is not None else ""}
                {f"<p><strong>Side View:</strong></p><img src='cid:side_view' alt='Side View'>" if alert.side_snapshot is not None else ""}
                {"</div>" if alert.front_snapshot is not None or alert.side_snapshot is not None else ""}
                
                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #868e96; font-size: 12px;">
                    <p>This is an automated alert from the Patient Contact Tracing System.</p>
                    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def _get_risk_class(self, risk_percent: float) -> str:
        """Get CSS class based on risk level."""
        if risk_percent >= 70:
            return "risk-high"
        elif risk_percent >= 40:
            return "risk-medium"
        else:
            return "risk-low"
    
    def _attach_snapshot(self, msg: MIMEMultipart, frame: np.ndarray, cid: str, name: str) -> None:
        """Attach camera snapshot to email."""
        try:
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                return
            
            img = MIMEImage(buffer.tobytes())
            img.add_header('Content-ID', f'<{cid}>')
            img.add_header('Content-Disposition', 'inline', filename=f'{name}.jpg')
            msg.attach(img)
        except Exception as e:
            rprint(f"[yellow]Warning: Failed to attach snapshot {name}: {e}[/]")

    # MongoDB query methods for the API
    def get_all_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all MDR alerts from MongoDB: mdr_marked (with past contacts) and mdr_contact (live)."""
        alerts = []
        
        # Get mdr_marked alerts (patient marked as MDR - includes past contacts)
        marked_alerts = list(self.collection.find(
            {"alert_type": "mdr_marked"}
        ).sort("timestamp", -1).limit(limit))
        
        for doc in marked_alerts:
            past_contacts = doc.get("past_contacts", [])
            alerts.append({
                "id": str(doc["_id"]),
                "alert_type": "mdr_marked",
                "mdr_patient": doc.get("mdr_patient"),
                "pathogen_type": doc.get("pathogen_type"),
                "pathogen_factor": doc.get("pathogen_factor"),
                "marked_by": doc.get("marked_by"),
                "notes": doc.get("notes"),
                "past_contacts": past_contacts,
                "past_contacts_count": len(past_contacts),
                "timestamp": doc.get("timestamp"),
                "created_at": doc.get("created_at"),
                "is_read": doc.get("read", False),
                "read": doc.get("read", False),
                "email_sent": doc.get("email_sent", False),
            })
        
        # Get mdr_contact alerts (live real-time contacts)
        pipeline = [
            {"$match": {"alert_type": "mdr_contact"}},
            {"$group": {
                "_id": {"mdr_patient": "$mdr_patient", "contacted_person": "$contacted_person"},
                "contact_count": {"$sum": 1},
                "total_duration": {"$sum": "$duration_seconds"},
                "max_risk": {"$max": "$risk_percent"},
                "min_distance": {"$min": "$min_distance_meters"},
                "last_distance": {"$last": "$distance_meters"},
                "first_contact": {"$min": "$created_at"},
                "last_contact": {"$max": "$created_at"},
                "last_id": {"$last": "$_id"},
                "any_email_sent": {"$max": {"$cond": ["$email_sent", 1, 0]}},
                "any_unread": {"$max": {"$cond": [{"$eq": ["$read", False]}, 1, 0]}},
                "has_snapshot": {"$max": {"$cond": [{"$or": [
                    {"$ifNull": ["$front_snapshot_base64", False]},
                    {"$ifNull": ["$side_snapshot_base64", False]}
                ]}, 1, 0]}}
            }},
            {"$sort": {"last_contact": -1}},
            {"$limit": limit}
        ]
        
        for doc in self.collection.aggregate(pipeline):
            alerts.append({
                "id": str(doc["last_id"]),
                "alert_type": "mdr_contact",
                "mdr_patient": doc["_id"]["mdr_patient"],
                "contacted_person": doc["_id"]["contacted_person"],
                "contact_count": doc["contact_count"],
                "duration_seconds": round(doc["total_duration"], 1),
                "duration_minutes": round(doc["total_duration"] / 60.0, 2),
                "risk_percent": round(doc["max_risk"], 1),
                "min_distance_meters": doc.get("min_distance"),
                "distance_meters": doc.get("last_distance"),
                "first_contact": doc["first_contact"],
                "created_at": doc["last_contact"],
                "timestamp": doc["last_contact"],  # When contact was detected
                "is_read": doc["any_unread"] == 0,
                "read": doc["any_unread"] == 0,
                "email_sent": doc["any_email_sent"] == 1,
                "has_front_snapshot": doc["has_snapshot"] == 1,
                "has_side_snapshot": doc["has_snapshot"] == 1
            })
        
        # Sort all alerts by timestamp
        alerts.sort(key=lambda x: x.get("timestamp") or x.get("created_at") or datetime.min, reverse=True)
        return alerts[:limit]

    def get_unread_alerts(self) -> List[Dict[str, Any]]:
        """Get unread MDR alerts including mdr_marked, mdr_contact, and backtrack types."""
        alerts = []
        
        # Get unread mdr_marked alerts
        # Get unread mdr_marked alerts (combined alerts with past contacts)
        marked_alerts = list(self.collection.find(
            {"alert_type": "mdr_marked", "read": False}
        ).sort("timestamp", -1))
        
        for doc in marked_alerts:
            alert_data = {
                "id": str(doc["_id"]),
                "alert_type": "mdr_marked",
                "mdr_patient": doc.get("mdr_patient"),
                "pathogen_type": doc.get("pathogen_type"),
                "marked_by": doc.get("marked_by"),
                "timestamp": doc.get("timestamp"),
                "created_at": doc.get("created_at"),
                "is_read": False,
                "read": False
            }
            # Include past contacts if available (combined alert format)
            if "past_contacts" in doc:
                alert_data["past_contacts"] = doc.get("past_contacts", [])
                alert_data["contact_count"] = len(doc.get("past_contacts", []))
            alerts.append(alert_data)
        
        # Get unread contact alerts (real-time future contacts)
        pipeline = [
            {"$match": {"alert_type": "mdr_contact", "read": False}},
            {"$group": {
                "_id": {"mdr_patient": "$mdr_patient", "contacted_person": "$contacted_person"},
                "contact_count": {"$sum": 1},
                "total_duration": {"$sum": "$duration_seconds"},
                "max_risk": {"$max": "$risk_percent"},
                "last_contact": {"$max": "$created_at"},
                "alert_created_at": {"$max": "$alert_created_at"},
                "alert_type": {"$last": "$alert_type"},
                "last_id": {"$last": "$_id"}
            }},
            {"$sort": {"alert_created_at": -1, "last_contact": -1}}
        ]
        
        for doc in self.collection.aggregate(pipeline):
            # Use alert_created_at (when alert was triggered) as display timestamp
            display_timestamp = doc.get("alert_created_at") or doc["last_contact"]
            alerts.append({
                "id": str(doc["last_id"]),
                "alert_type": "mdr_contact",
                "mdr_patient": doc["_id"]["mdr_patient"],
                "contacted_person": doc["_id"]["contacted_person"],
                "contact_count": doc["contact_count"],
                "duration_seconds": round(doc["total_duration"], 1),
                "duration_minutes": round(doc["total_duration"] / 60.0, 2),
                "risk_percent": round(doc["max_risk"], 1),
                "created_at": doc["last_contact"],
                "timestamp": display_timestamp,
                "is_read": False,
                "read": False
            })
        
        alerts.sort(key=lambda x: x.get("timestamp") or x.get("created_at") or datetime.min, reverse=True)
        return alerts

    def get_unread_count(self) -> int:
        """Get count of unique unread alerts (mdr_marked + contact pairs)."""
        # Count unread mdr_marked alerts
        marked_count = self.collection.count_documents({"alert_type": "mdr_marked", "read": False})
        
        # Count unique unread contact pairs (real-time future contacts)
        pipeline = [
            {"$match": {"alert_type": "mdr_contact", "read": False}},
            {"$group": {
                "_id": {"mdr_patient": "$mdr_patient", "contacted_person": "$contacted_person"}
            }},
            {"$count": "total"}
        ]
        result = list(self.collection.aggregate(pipeline))
        contact_count = result[0]["total"] if result else 0
        
        return marked_count + contact_count

    def mark_as_read(self, alert_id: str) -> bool:
        """Mark alert as read. For contact alerts, mark all for the same pair."""
        from bson import ObjectId
        
        # First, get the alert to find the type
        alert = self.collection.find_one({"_id": ObjectId(alert_id)})
        if not alert:
            return False
        
        # For mdr_marked alerts, just mark that one
        if alert.get("alert_type") == "mdr_marked":
            result = self.collection.update_one(
                {"_id": ObjectId(alert_id)},
                {"$set": {"read": True, "read_at": datetime.utcnow()}}
            )
            return result.modified_count > 0
        
        # For contact alerts, mark all alerts for this pair as read
        result = self.collection.update_many(
            {
                "alert_type": "mdr_contact",
                "mdr_patient": alert["mdr_patient"],
                "contacted_person": alert["contacted_person"]
            },
            {"$set": {"read": True, "read_at": datetime.utcnow()}}
        )
        return result.modified_count > 0

    def mark_all_as_read(self) -> int:
        """Mark all alerts as read. Returns count of updated documents."""
        result = self.collection.update_many(
            {"alert_type": {"$in": ["mdr_contact", "mdr_marked"]}, "read": False},
            {"$set": {"read": True, "read_at": datetime.utcnow()}}
        )
        return result.modified_count

    def get_alert_detail(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed alert with snapshots."""
        from bson import ObjectId
        doc = self.collection.find_one({"_id": ObjectId(alert_id)})
        if not doc:
            return None
        
        # Use contact_start for display timestamp (local time)
        display_timestamp = doc.get("contact_start") or doc["created_at"]
        
        return {
            "id": str(doc["_id"]),
            "mdr_patient": doc["mdr_patient"],
            "contacted_person": doc["contacted_person"],
            "contact_start": doc["contact_start"],
            "contact_end": doc.get("contact_end"),
            "duration_seconds": doc["duration_seconds"],
            "duration_minutes": doc["duration_seconds"] / 60.0,
            "risk_percent": doc["risk_percent"],
            "status": doc.get("status", "completed"),
            "created_at": doc["created_at"],
            "timestamp": display_timestamp,  # Use local time from contact_start
            "read": doc.get("read", False),
            "email_sent": doc.get("email_sent", False),
            "front_snapshot_base64": doc.get("front_snapshot_base64"),
            "side_snapshot_base64": doc.get("side_snapshot_base64")
        }

    def get_alerts_for_patient(self, patient_name: str) -> List[Dict[str, Any]]:
        """Get all alerts involving a specific MDR patient."""
        alerts = []
        for doc in self.collection.find({
            "alert_type": "mdr_contact",
            "mdr_patient": patient_name
        }).sort("created_at", -1):
            alerts.append({
                "id": str(doc["_id"]),
                "contacted_person": doc["contacted_person"],
                "contact_start": doc["contact_start"],
                "contact_end": doc.get("contact_end"),
                "duration_minutes": doc["duration_seconds"] / 60.0,
                "risk_percent": doc["risk_percent"],
                "created_at": doc["created_at"]
            })
        return alerts


# Singleton instance
_email_alerter_instance: Optional[EmailAlerterMongo] = None


def get_email_alerter() -> EmailAlerterMongo:
    """Get or create the email alerter instance."""
    global _email_alerter_instance
    if _email_alerter_instance is None:
        _email_alerter_instance = EmailAlerterMongo()
    return _email_alerter_instance


__all__ = ["EmailAlerterMongo", "MDRContactAlert", "get_email_alerter"]
