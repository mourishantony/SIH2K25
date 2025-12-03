"""Email alerting system for MDR patient contact notifications."""
from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from rich import print as rprint

from config import ROOT_DIR

load_dotenv(ROOT_DIR / ".env", override=False)


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


class EmailAlerter:
    """Send email alerts for MDR patient contacts."""
    
    def __init__(self):
        self.smtp_server = os.getenv("MDR_SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("MDR_SMTP_PORT", "587"))
        self.smtp_username = os.getenv("MDR_SMTP_USERNAME", "")
        self.smtp_password = os.getenv("MDR_SMTP_PASSWORD", "")
        self.admin_email = os.getenv("MDR_ADMIN_EMAIL", "")
        self.from_email = os.getenv("MDR_FROM_EMAIL", self.smtp_username)
        self.enabled = bool(self.smtp_username and self.smtp_password and self.admin_email)
        
        if not self.enabled:
            rprint("[yellow]⚠ MDR email alerts disabled: missing SMTP credentials in .env[/]")
    
    def send_mdr_alert(self, alert: MDRContactAlert) -> bool:
        """Send MDR contact alert email to admin. Returns True if successful."""
        if not self.enabled:
            rprint("[yellow]MDR email alert skipped (not configured)[/]")
            return False
        
        try:
            msg = self._create_email(alert)
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            rprint(f"[bold green]✓[/] MDR alert email sent to {self.admin_email}")
            return True
            
        except Exception as e:
            rprint(f"[bold red]✗[/] Failed to send MDR alert email: {e}")
            return False
    
    def _create_email(self, alert: MDRContactAlert) -> MIMEMultipart:
        """Create the email message with alert details and snapshots."""
        msg = MIMEMultipart("related")
        msg["Subject"] = f"🚨 MDR Patient Contact Alert - {alert.mdr_patient}"
        msg["From"] = self.from_email
        msg["To"] = self.admin_email
        
        # Create HTML body
        html_body = self._create_html_body(alert)
        msg.attach(MIMEText(html_body, "html"))
        
        # Attach snapshots if available
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
            # Encode frame as JPEG
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                return
            
            img = MIMEImage(buffer.tobytes())
            img.add_header('Content-ID', f'<{cid}>')
            img.add_header('Content-Disposition', 'inline', filename=f'{name}.jpg')
            msg.attach(img)
        except Exception as e:
            rprint(f"[yellow]Warning: Failed to attach snapshot {name}: {e}[/]")
