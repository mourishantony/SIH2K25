from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console
from rich.text import Text

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


class AlertSystem:
    def __init__(
        self,
        *,
        log_dir: Path,
        min_risk: float = 0.4,
        duration_threshold: float = 10.0,
        min_alert_interval: float = 5.0,
        enable_logging: bool = True,
        enable_audio: bool = False,
    ) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.min_risk = min_risk
        self.duration_threshold = duration_threshold
        self.min_alert_interval = min_alert_interval
        self.enable_logging = enable_logging
        self.enable_audio = enable_audio
        self.recent_alerts: Dict[tuple[str, str], float] = {}

    def should_alert(self, collision) -> bool:
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
    ) -> None:
        active = collision_cam1 if collision_cam1 is not None else collision_cam2
        if active is None:
            return
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
            self._log_alert(event)

    def _log_alert(self, event: AlertEvent) -> None:
        day_file = self.log_dir / f"alerts_{datetime.utcnow().date()}.json"
        payload = []
        if day_file.exists():
            try:
                payload = json.loads(day_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = []
        payload.append(asdict(event))
        day_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _beep(self) -> None:
        try:
            import winsound

            winsound.Beep(1200, 250)
        except Exception:
            pass


__all__ = ["AlertSystem", "AlertEvent"]
