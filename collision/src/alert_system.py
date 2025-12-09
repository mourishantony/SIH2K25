from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rich import print as rprint
from rich.console import Console
from rich.table import Table

from collision_detector import Collision

ALERT_LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "alerts"


@dataclass
class AlertEvent:
    timestamp: str
    person1: str
    person2: str
    risk_level: str
    risk_score: float
    camera1_iou: float
    camera2_iou: Optional[float]
    verified_by_both_cameras: bool
    frame_number: int
    collision_duration: float = 0.0
    distance: float = 0.0
    frame_count: int = 0
    start_time: float = 0.0

    def to_dict(self):
        return asdict(self)


class AlertSystem:

    def __init__(
        self,
        enable_audio: bool = False,
        enable_logging: bool = True,
        min_alert_interval: float = 2.0,
        min_risk_for_alert: str = "MEDIUM",
        min_collision_duration: float = 10.0,
        min_risk_score: float = 0.30,
        high_risk_threshold: float = 0.40,
        enable_high_risk_json: bool = True,
    ):
        self.enable_audio = enable_audio
        self.enable_logging = enable_logging
        self.min_alert_interval = min_alert_interval
        self.min_risk_for_alert = min_risk_for_alert
        self.min_collision_duration = min_collision_duration
        self.min_risk_score = min_risk_score
        self.high_risk_threshold = high_risk_threshold
        self.enable_high_risk_json = enable_high_risk_json
        
        self.recent_alerts: dict = {}
        
        self.collision_start_times: dict = {}
        
        self.alert_count = 0
        
        self.console = Console()
        
        if enable_logging:
            ALERT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        self.risk_hierarchy = {
            "SAFE": 0,
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
            "CRITICAL": 4,
        }

    def should_alert(self, collision: Collision) -> bool:
        if hasattr(collision, 'duration') and collision.duration < self.min_collision_duration:
            return False
        
        if collision.risk_score < self.min_risk_score:
            return False
        
        if hasattr(collision, 'risk_level') and collision.risk_level in self.risk_hierarchy:
            min_level = self.risk_hierarchy.get(self.min_risk_for_alert, 2)
            collision_level = self.risk_hierarchy.get(collision.risk_level, 0)
            if collision_level < min_level:
                return False
        
        people = tuple(sorted([collision.person1, collision.person2]))
        
        current_time = time.time()
        if people in self.recent_alerts:
            last_alert_time = self.recent_alerts[people]
            if current_time - last_alert_time < self.min_alert_interval:
                return False
        
        self.recent_alerts[people] = current_time
        return True
    
    def clear_collision_tracking(self, collision: Collision):
        people = tuple(sorted([collision.person1, collision.person2]))
        if people in self.collision_start_times:
            del self.collision_start_times[people]

    def save_high_risk_collision_json(self, collision: Collision, frame_number: int):
        if collision.risk_score < self.high_risk_threshold:
            return
        
        collision_data = {
            "timestamp": datetime.now().isoformat(),
            "collision_id": collision.get_collision_id(),
            "person1": collision.person1,
            "person2": collision.person2,
            "risk_level": collision.risk_level,
            "risk_score": round(float(collision.risk_score), 3),
            "duration_seconds": round(float(collision.duration), 1),
            "frame_count": int(collision.frame_count),
            "iou": round(float(collision.iou), 3),
            "distance_pixels": round(float(collision.distance), 1),
            "frame_number": int(frame_number),
            "bbox1": {
                "x1": int(collision.bbox1.x1),
                "y1": int(collision.bbox1.y1),
                "x2": int(collision.bbox1.x2),
                "y2": int(collision.bbox1.y2),
                "confidence": round(float(collision.bbox1.confidence), 3)
            },
            "bbox2": {
                "x1": int(collision.bbox2.x1),
                "y1": int(collision.bbox2.y1),
                "x2": int(collision.bbox2.x2),
                "y2": int(collision.bbox2.y2),
                "confidence": round(float(collision.bbox2.confidence), 3)
            }
        }
        
        json_file = ALERT_LOG_DIR / f"high_risk_collisions_{datetime.now().strftime('%Y-%m-%d')}.json"
        
        if json_file.exists():
            with open(json_file, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        
        collision_id = collision_data["collision_id"]
        recent_entries = [d for d in data if d.get("collision_id") == collision_id]
        
        should_add = True
        if recent_entries:
            latest_entry = max(recent_entries, key=lambda x: x.get("timestamp", ""))
            try:
                latest_time = datetime.fromisoformat(latest_entry["timestamp"])
                current_time = datetime.fromisoformat(collision_data["timestamp"])
                if (current_time - latest_time).total_seconds() < 5.0:
                    should_add = False
            except:
                pass
        
        if should_add:
            data.append(collision_data)
            
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            rprint(f"[yellow]💾 High-risk collision saved to JSON: {collision.get_collision_id()} (risk: {collision.risk_score:.1%}, duration: {collision.duration:.1f}s)[/yellow]")
    
    def trigger_alert(
        self,
        collision_cam1: Collision,
        collision_cam2: Optional[Collision],
        frame_number: int,
        verified: bool,
    ):
        self.alert_count += 1
        
        collision = collision_cam1 if collision_cam1 else collision_cam2
        
        if not collision:
            return
        
        event = AlertEvent(
            timestamp=datetime.now().isoformat(),
            person1=collision.person1,
            person2=collision.person2,
            risk_level=collision.risk_level,
            risk_score=collision.risk_score,
            camera1_iou=collision_cam1.iou if collision_cam1 else None,
            camera2_iou=collision_cam2.iou if collision_cam2 else None,
            verified_by_both_cameras=verified,
            frame_number=frame_number,
            collision_duration=0.0,
        )
        
        self._display_alert(event)
        
        if self.enable_logging:
            self._log_alert(event)
        
        if self.enable_audio:
            self._play_audio_alert(event.risk_level)

    def _display_alert(self, event: AlertEvent):
        color_map = {
            "CRITICAL": "bold red on white",
            "HIGH": "bold red",
            "MEDIUM": "bold yellow",
            "LOW": "yellow",
            "SAFE": "green",
        }
        color = color_map.get(event.risk_level, "white")
        
        verified_text = "✓ Collision in BOTH camera frames"
        
        self.console.print()
        self.console.print(f"[{color}]{'='*70}[/]")
        self.console.print(f"[{color}]⚠️  COLLISION ALERT #{self.alert_count} - {event.risk_level} RISK[/]")
        self.console.print(f"[{color}]{'='*70}[/]")
        self.console.print(f"  Time: {event.timestamp}")
        self.console.print(f"  People: {event.person1} <-> {event.person2}")
        self.console.print(f"  Risk Score: {event.risk_score:.2%}")
        if event.camera1_iou is not None:
            self.console.print(f"  Camera 1 IoU: {event.camera1_iou:.3f}")
        if event.camera2_iou is not None:
            self.console.print(f"  Camera 2 IoU: {event.camera2_iou:.3f}")
        self.console.print(f"  Status: {verified_text}")
        self.console.print(f"  Frame: {event.frame_number}")
        self.console.print(f"[{color}]{'='*70}[/]")
        self.console.print()

    def _log_alert(self, event: AlertEvent):
        log_date = datetime.now().strftime("%Y-%m-%d")
        log_file = ALERT_LOG_DIR / f"alerts_{log_date}.json"
        
        alerts = []
        if log_file.exists():
            with log_file.open("r", encoding="utf-8") as f:
                alerts = json.load(f)
        
        alerts.append(event.to_dict())
        
        with log_file.open("w", encoding="utf-8") as f:
            json.dump(alerts, f, indent=2)

    def _play_audio_alert(self, risk_level: str):
        try:
            import winsound
            freq_map = {
                "CRITICAL": 2000,
                "HIGH": 1500,
                "MEDIUM": 1000,
                "LOW": 800,
            }
            frequency = freq_map.get(risk_level, 1000)
            duration = 200
            winsound.Beep(frequency, duration)
        except (ImportError, RuntimeError):
            pass

    def display_summary_table(self, collisions_cam1: List[Collision], collisions_cam2: List[Collision]):
        if not collisions_cam1 and not collisions_cam2:
            return
        
        table = Table(title="Active Collision Detections")
        table.add_column("Camera", style="cyan")
        table.add_column("Person 1", style="magenta")
        table.add_column("Person 2", style="magenta")
        table.add_column("Risk Level", style="bold")
        table.add_column("Score", justify="right")
        table.add_column("IoU", justify="right")
        table.add_column("Distance", justify="right")
        
        for collision in collisions_cam1:
            color = self._get_risk_color(collision.risk_level)
            table.add_row(
                "Cam 1",
                collision.person1,
                collision.person2,
                f"[{color}]{collision.risk_level}[/]",
                f"{collision.risk_score:.2f}",
                f"{collision.iou:.3f}",
                f"{collision.distance:.0f}px",
            )
        
        for collision in collisions_cam2:
            color = self._get_risk_color(collision.risk_level)
            table.add_row(
                "Cam 2",
                collision.person1,
                collision.person2,
                f"[{color}]{collision.risk_level}[/]",
                f"{collision.risk_score:.2f}",
                f"{collision.iou:.3f}",
                f"{collision.distance:.0f}px",
            )
        
        self.console.print(table)

    def _get_risk_color(self, risk_level: str) -> str:
        color_map = {
            "CRITICAL": "bold red",
            "HIGH": "red",
            "MEDIUM": "yellow",
            "LOW": "blue",
            "SAFE": "green",
        }
        return color_map.get(risk_level, "white")

    def get_statistics(self) -> dict:
        return {
            "total_alerts": self.alert_count,
            "active_tracked_pairs": len(self.recent_alerts),
        }
