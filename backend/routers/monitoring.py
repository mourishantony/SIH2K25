"""AI Monitoring router for video processing and live stream contact tracing."""
from __future__ import annotations

import os
import uuid
import asyncio
import base64
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from routers.auth import get_current_user, require_permission

router = APIRouter()

# Get the root directory and video uploads path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
VIDEO_UPLOAD_DIR = ROOT_DIR / "data" / "video_uploads"
VIDEO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class MonitoringStatus(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class MonitoringMode(str, Enum):
    VIDEO = "video"
    WEBCAM = "webcam"


class MonitoringConfig(BaseModel):
    mode: MonitoringMode
    front_video_path: Optional[str] = None
    side_video_path: Optional[str] = None
    front_camera_index: int = 0
    side_camera_index: int = 1
    use_gpu: bool = False
    min_confidence: float = 0.35
    threshold: float = 0.32
    base_rate: float = 0.02
    event_penalty: float = 0.05


class MonitoringSession:
    """Tracks the current monitoring session state."""
    
    def __init__(self):
        self.status = MonitoringStatus.IDLE
        self.mode: Optional[MonitoringMode] = None
        self.config: Optional[MonitoringConfig] = None
        self.session_id: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.started_by: Optional[str] = None
        self.error_message: Optional[str] = None
        self.stats: Dict[str, Any] = {}
        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._websocket_clients: List[WebSocket] = []
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=30)
        
    def reset(self):
        self.status = MonitoringStatus.IDLE
        self.mode = None
        self.config = None
        self.session_id = None
        self.started_at = None
        self.started_by = None
        self.error_message = None
        self.stats = {}
        self._task = None
        self._stop_event = None
        self._frame_queue = asyncio.Queue(maxsize=30)

    async def broadcast_frame(self, frame_data: dict):
        """Send frame to all connected WebSocket clients."""
        disconnected = []
        for ws in self._websocket_clients:
            try:
                await ws.send_json(frame_data)
            except Exception:
                disconnected.append(ws)
        
        # Remove disconnected clients
        for ws in disconnected:
            self._websocket_clients.remove(ws)

    async def broadcast_status(self, status_data: dict):
        """Send status update to all connected WebSocket clients."""
        await self.broadcast_frame({"type": "status", **status_data})


# Global monitoring session (single session at a time)
_monitoring_session = MonitoringSession()


def get_monitoring_session() -> MonitoringSession:
    return _monitoring_session


# ============================================
# REST API Endpoints
# ============================================

@router.get("/status")
async def get_monitoring_status(
    current_user: dict = Depends(require_permission("monitoring"))
):
    """Get current monitoring session status."""
    session = get_monitoring_session()
    
    return {
        "status": session.status.value,
        "mode": session.mode.value if session.mode else None,
        "session_id": session.session_id,
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "started_by": session.started_by,
        "error_message": session.error_message,
        "stats": session.stats,
        "connected_clients": len(session._websocket_clients)
    }


@router.post("/upload-video")
async def upload_video(
    video_type: str = Form(..., description="'front' or 'side'"),
    video_file: UploadFile = File(...),
    current_user: dict = Depends(require_permission("monitoring"))
):
    """Upload a video file for processing."""
    if video_type not in ["front", "side"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="video_type must be 'front' or 'side'"
        )
    
    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    file_ext = Path(video_file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    new_filename = f"{video_type}_{timestamp}_{unique_id}{file_ext}"
    file_path = VIDEO_UPLOAD_DIR / new_filename
    
    # Save file
    try:
        with open(file_path, "wb") as f:
            content = await video_file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save video: {str(e)}"
        )
    
    return {
        "message": f"Video uploaded successfully",
        "video_type": video_type,
        "filename": new_filename,
        "path": str(file_path),
        "size_bytes": file_path.stat().st_size
    }


@router.get("/uploaded-videos")
async def list_uploaded_videos(
    current_user: dict = Depends(require_permission("monitoring"))
):
    """List all uploaded videos."""
    videos = []
    
    for file_path in VIDEO_UPLOAD_DIR.glob("*"):
        if file_path.is_file() and file_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            name = file_path.name
            video_type = "front" if name.startswith("front_") else "side" if name.startswith("side_") else "unknown"
            
            videos.append({
                "filename": name,
                "path": str(file_path),
                "video_type": video_type,
                "size_bytes": file_path.stat().st_size,
                "created_at": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()
            })
    
    # Sort by creation time, newest first
    videos.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "total": len(videos),
        "videos": videos
    }


@router.delete("/uploaded-videos/{filename}")
async def delete_uploaded_video(
    filename: str,
    current_user: dict = Depends(require_permission("monitoring"))
):
    """Delete an uploaded video file."""
    file_path = VIDEO_UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video file not found"
        )
    
    try:
        file_path.unlink()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete video: {str(e)}"
        )
    
    return {"message": f"Video '{filename}' deleted successfully"}


@router.post("/start")
async def start_monitoring(
    config: MonitoringConfig,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(require_permission("monitoring"))
):
    """Start a monitoring session."""
    session = get_monitoring_session()
    
    # Check if already running
    if session.status in [MonitoringStatus.RUNNING, MonitoringStatus.STARTING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Monitoring is already {session.status.value}. Stop it first."
        )
    
    # Validate config for video mode
    if config.mode == MonitoringMode.VIDEO:
        if not config.front_video_path or not config.side_video_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both front_video_path and side_video_path are required for video mode"
            )
        
        # Check files exist
        front_path = Path(config.front_video_path)
        side_path = Path(config.side_video_path)
        
        if not front_path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Front video file not found: {config.front_video_path}"
            )
        
        if not side_path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Side video file not found: {config.side_video_path}"
            )
    
    # Initialize session
    session.status = MonitoringStatus.STARTING
    session.mode = config.mode
    session.config = config
    session.session_id = str(uuid.uuid4())[:12]
    session.started_at = datetime.now()
    session.started_by = current_user.get("username", "unknown")
    session.error_message = None
    session._stop_event = asyncio.Event()
    
    # Start monitoring in background
    background_tasks.add_task(run_monitoring_session, session)
    
    return {
        "message": "Monitoring session starting",
        "session_id": session.session_id,
        "mode": config.mode.value
    }


@router.post("/stop")
async def stop_monitoring(
    current_user: dict = Depends(require_permission("monitoring"))
):
    """Stop the current monitoring session."""
    session = get_monitoring_session()
    
    if session.status not in [MonitoringStatus.RUNNING, MonitoringStatus.STARTING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No active monitoring session to stop"
        )
    
    session.status = MonitoringStatus.STOPPING
    
    # Signal the monitoring task to stop
    if session._stop_event:
        session._stop_event.set()
    
    # Broadcast stop status
    await session.broadcast_status({
        "status": "stopping",
        "message": "Monitoring session is stopping..."
    })
    
    return {
        "message": "Monitoring session stopping",
        "session_id": session.session_id
    }


@router.get("/config")
async def get_monitoring_config(
    current_user: dict = Depends(require_permission("monitoring"))
):
    """Get default monitoring configuration."""
    from config import recognition_settings, contact_settings, dual_view_settings
    
    return {
        "defaults": {
            "use_gpu": recognition_settings.use_gpu,
            "min_confidence": recognition_settings.min_confidence,
            "threshold": recognition_settings.threshold,
            "base_rate": contact_settings.base_rate,
            "event_penalty": contact_settings.event_penalty,
            "front_camera_index": dual_view_settings.front_camera_index,
            "side_camera_index": dual_view_settings.side_camera_index,
        }
    }


# ============================================
# WebSocket for real-time frames and status
# ============================================

@router.websocket("/ws")
async def monitoring_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring frames and status updates."""
    await websocket.accept()
    
    session = get_monitoring_session()
    session._websocket_clients.append(websocket)
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "status": session.status.value,
            "session_id": session.session_id,
            "mode": session.mode.value if session.mode else None,
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                
                # Handle ping
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                
                # Handle browser webcam frames
                elif data.get("type") == "webcam_frame":
                    # This receives frames from browser webcams
                    # Forward to the frame processing queue
                    view = data.get("view", "front")
                    frame_b64 = data.get("frame")
                    
                    if frame_b64 and session.status == MonitoringStatus.RUNNING:
                        try:
                            # Put frame in queue for processing
                            await asyncio.wait_for(
                                session._frame_queue.put({
                                    "view": view,
                                    "frame": frame_b64,
                                    "timestamp": datetime.now().isoformat()
                                }),
                                timeout=0.1
                            )
                        except asyncio.TimeoutError:
                            # Queue full, skip frame
                            pass
                    
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive"})
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        if websocket in session._websocket_clients:
            session._websocket_clients.remove(websocket)


# ============================================
# Background monitoring task
# ============================================

async def run_monitoring_session(session: MonitoringSession):
    """Background task to run the monitoring process."""
    try:
        session.status = MonitoringStatus.RUNNING
        await session.broadcast_status({
            "status": "running",
            "message": "Monitoring session started",
            "session_id": session.session_id
        })
        
        # Add src directory to path for imports
        import sys
        src_dir = str(ROOT_DIR / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        
        # Import and run the monitor service
        from monitor_service import ContactMonitorService
        
        # Create service instance
        service = ContactMonitorService(
            mode=session.config.mode.value,
            front_video_path=session.config.front_video_path,
            side_video_path=session.config.side_video_path,
            front_camera_index=session.config.front_camera_index,
            side_camera_index=session.config.side_camera_index,
            use_gpu=session.config.use_gpu,
            min_confidence=session.config.min_confidence,
            threshold=session.config.threshold,
            base_rate=session.config.base_rate,
            event_penalty=session.config.event_penalty,
        )
        
        # Run the monitoring loop
        async for frame_data in service.run_async(session._stop_event):
            if session._stop_event and session._stop_event.is_set():
                break
            
            # Broadcast frame to connected clients
            await session.broadcast_frame({
                "type": "frame",
                **frame_data
            })
            
            # Update stats
            session.stats = frame_data.get("stats", {})
        
        # Clean shutdown
        service.cleanup()
        
    except Exception as e:
        session.status = MonitoringStatus.ERROR
        session.error_message = str(e)
        await session.broadcast_status({
            "status": "error",
            "message": f"Monitoring error: {str(e)}"
        })
        print(f"[Monitoring] Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        session.status = MonitoringStatus.IDLE
        await session.broadcast_status({
            "status": "idle",
            "message": "Monitoring session ended"
        })
        session.reset()
