"""Alert system with MongoDB storage."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional

from rich.console import Console
from rich.text import Text

from database import get_collision_alerts_collection, get_alerts_collection

console = Console()


@dataclass
class AlertEvent:
    timestamp: str
    person1: str
    person2: str
    risk_level: str
    risk_score: float
    camera1_iou: Optional[float]
    camera2_iou: Optional[float]
    verified_by_both_cameras: bool
    frame_number: int


class AlertSystemMongo:
    """Alert system that stores alerts in MongoDB."""

    def __init__(
        self,
        *,
        min_risk: float = 0.4,
        duration_threshold: float = 10.0,
        min_alert_interval: float = 5.0,
        enable_logging: bool = True,
        enable_audio: bool = False,
    ) -> None:
        self.min_risk = min_risk
        self.duration_threshold = duration_threshold
        self.min_alert_interval = min_alert_interval
        self.enable_logging = enable_logging
        self.enable_audio = enable_audio
        self.recent_alerts: Dict[tuple, float] = {}
        self.collection = get_collision_alerts_collection()

    def should_alert(self, collision) -> bool:
        """Check if an alert should be triggered."""
        if collision is None:
            return False
        if getattr(collision, "duration", 0.0) < self.duration_threshold:
            return False
        if collision.risk_score < self.min_risk:
            return False
        
        pair = tuple(sorted((collision.person1, collision.person2)))
        now = time.time()
        last = self.recent_alerts.get(pair)
        
        if last is not None and (now - last) < self.min_alert_interval:
            return False
        
        self.recent_alerts[pair] = now
        return True

    def trigger_alert(
        self,
        collision_cam1,
        collision_cam2,
        *,
        frame_number: int,
        verified: bool,
    ) -> Optional[str]:
        """Trigger an alert and store in MongoDB. Returns document ID."""
        active = collision_cam1 if collision_cam1 is not None else collision_cam2
        if active is None:
            return None
        
        # Console output
        message = Text()
        message.append("[COLLISION] ", style="bold red")
        message.append(f"{active.person1} ↔ {active.person2} | ")
        message.append(f"Risk: {active.risk_level} ({active.risk_score:.2f}) | ")
        message.append(f"Duration: {getattr(active, 'duration', 0.0):.1f}s")
        console.print(message)
        
        if self.enable_audio:
            self._beep()
        
        if self.enable_logging:
            event = AlertEvent(
                timestamp=datetime.utcnow().isoformat(),
                person1=active.person1,
                person2=active.person2,
                risk_level=active.risk_level,
                risk_score=active.risk_score,
                camera1_iou=collision_cam1.iou if collision_cam1 else None,
                camera2_iou=collision_cam2.iou if collision_cam2 else None,
                verified_by_both_cameras=verified,
                frame_number=frame_number,
            )
            return self._log_alert(event)
        
        return None

    def _log_alert(self, event: AlertEvent) -> str:
        """Log alert to MongoDB. Returns document ID."""
        doc = asdict(event)
        doc["created_at"] = datetime.utcnow()
        doc["read"] = False
        result = self.collection.insert_one(doc)
        return str(result.inserted_id)

    def _beep(self) -> None:
        """Play alert sound on Windows."""
        try:
            import winsound
            winsound.Beep(1200, 250)
        except Exception:
            pass

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent collision alerts from MongoDB."""
        alerts = []
        for doc in self.collection.find().sort("created_at", -1).limit(limit):
            alerts.append({
                "id": str(doc["_id"]),
                "timestamp": doc["timestamp"],
                "person1": doc["person1"],
                "person2": doc["person2"],
                "risk_level": doc["risk_level"],
                "risk_score": doc["risk_score"],
                "verified": doc.get("verified_by_both_cameras", False),
                "read": doc.get("read", False),
                "created_at": doc.get("created_at")
            })
        return alerts

    def mark_alert_read(self, alert_id: str) -> bool:
        """Mark an alert as read."""
        from bson import ObjectId
        result = self.collection.update_one(
            {"_id": ObjectId(alert_id)},
            {"$set": {"read": True}}
        )
        return result.modified_count > 0

    def get_unread_count(self) -> int:
        """Get count of unread collision alerts."""
        return self.collection.count_documents({"read": False})


# Singleton instance
_alert_system_instance: Optional[AlertSystemMongo] = None


def get_alert_system(
    min_risk: float = 0.4,
    duration_threshold: float = 10.0,
    min_alert_interval: float = 5.0,
    enable_audio: bool = False,
) -> AlertSystemMongo:
    """Get or create the alert system instance."""
    global _alert_system_instance
    if _alert_system_instance is None:
        _alert_system_instance = AlertSystemMongo(
            min_risk=min_risk,
            duration_threshold=duration_threshold,
            min_alert_interval=min_alert_interval,
            enable_audio=enable_audio,
        )
    return _alert_system_instance


__all__ = ["AlertSystemMongo", "AlertEvent", "get_alert_system"]
