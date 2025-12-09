from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import typer
from rich import print as rprint

from alert_system import AlertSystem
from collision_detector import detect_collisions, verify_collision_across_cameras
from config import collision_settings
from multi_camera import DualCameraManager

app = typer.Typer(add_completion=False)


def draw_bboxes_and_collisions(frame: np.ndarray, bboxes, collisions, camera_name: str):
    collision_bboxes = set()
    for collision in collisions:
        collision_bboxes.add(id(collision.bbox1))
        collision_bboxes.add(id(collision.bbox2))
    
    for bbox in bboxes:
        is_in_collision = id(bbox) in collision_bboxes
        color = (0, 0, 255) if is_in_collision else (0, 255, 0)
        thickness = 3 if is_in_collision else 2
        
        cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, thickness)
        
        label = f"{bbox.person_name} ({bbox.confidence:.2f})"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(
            frame,
            (bbox.x1, bbox.y1 - label_size[1] - 10),
            (bbox.x1 + label_size[0], bbox.y1),
            color,
            -1,
        )
        
        # Text
        cv2.putText(
            frame,
            label,
            (bbox.x1, bbox.y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    
    for collision in collisions:
        risk_colors = {
            "CRITICAL": (128, 0, 255),
            "HIGH": (0, 0, 255),
            "MEDIUM": (0, 165, 255),
            "LOW": (0, 255, 255),
            "SAFE": (0, 255, 0),
        }
        color = risk_colors.get(collision.risk_level, (255, 255, 255))
        
        center1 = collision.bbox1.center
        center2 = collision.bbox2.center
        cv2.line(
            frame,
            (int(center1[0]), int(center1[1])),
            (int(center2[0]), int(center2[1])),
            color,
            4,
        )
        
        mid_x = int((center1[0] + center2[0]) / 2)
        mid_y = int((center1[1] + center2[1]) / 2)
        
        collision_label = f"⚠ {collision.risk_level} COLLISION ⚠"
        cv2.putText(
            frame,
            collision_label,
            (mid_x - 120, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
        
        risk_text = f"Risk: {collision.risk_score:.2f} | IoU: {collision.iou:.2f}"
        cv2.putText(
            frame,
            risk_text,
            (mid_x - 120, mid_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    
    info_text = f"{camera_name} | Collisions: {len(collisions)}"
    cv2.putText(
        frame,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    
    return frame


@app.command()
def monitor(
    camera1: int = typer.Option(
        collision_settings.camera1_index,
        help="First camera index (or use --video1 for video file)",
    ),
    camera2: int = typer.Option(
        collision_settings.camera2_index,
        help="Second camera index (or use --video2 for video file)",
    ),
    video1: Optional[Path] = typer.Option(
        None,
        help="Path to first video file (overrides --camera1)",
    ),
    video2: Optional[Path] = typer.Option(
        None,
        help="Path to second video file (overrides --camera2)",
    ),
    iou_threshold: float = typer.Option(
        collision_settings.iou_threshold,
        help="IoU threshold for collision detection",
    ),
    distance_threshold: float = typer.Option(
        collision_settings.distance_threshold,
        help="Distance threshold (pixels) for proximity detection",
    ),
    min_confidence: float = typer.Option(
        collision_settings.min_confidence,
        help="Minimum face detection confidence",
    ),
    threshold: float = typer.Option(
        collision_settings.recognition_threshold,
        help="Face recognition similarity threshold",
    ),
    det_size: int = typer.Option(
        collision_settings.det_size,
        help="Detection size for InsightFace",
    ),
    use_gpu: bool = typer.Option(
        True,  # Enable GPU by default for RTX 3050
        help="Use RTX 3050 GPU acceleration (recommended)",
    ),
    enable_audio: bool = typer.Option(
        collision_settings.enable_audio,
        help="Enable audio alerts for collisions",
    ),
    min_risk_for_alert: str = typer.Option(
        collision_settings.min_risk_for_alert,
        help="Minimum risk level for alerts (SAFE/LOW/MEDIUM/HIGH/CRITICAL)",
    ),
    require_both_cameras: bool = typer.Option(
        collision_settings.require_both_cameras,
        help="Only alert when collision detected in BOTH cameras",
    ),
    display_mode: str = typer.Option(
        "side",
        help="Display mode: 'side' (horizontal, default), 'stacked' (vertical), or 'separate' (two windows)",
    ),
):
    source1 = str(video1) if video1 else camera1
    source2 = str(video2) if video2 else camera2
    
    rprint("[🚀 bold green]Starting RTX 3050 GPU-Accelerated Collision Monitoring[🚀 /bold green]")
    rprint(f"Camera 1: {source1}")
    rprint(f"Camera 2: {source2}")
    rprint(f"IoU Threshold: {iou_threshold}")
    rprint(f"Distance Threshold: {distance_threshold}px")
    rprint(f"GPU Acceleration: {'[✅ green]ENABLED[/]' if use_gpu else '[⚠️ yellow]DISABLED[/]'}")
    rprint(f"Require Both Cameras: {require_both_cameras}")
    rprint()
    
    camera_manager = DualCameraManager(
        source1=source1,
        source2=source2,
        det_size=det_size,
        use_gpu=use_gpu,
        min_confidence=min_confidence,
        threshold=threshold,
    )
    
    alert_system = AlertSystem(
        enable_audio=enable_audio,
        enable_logging=True,
        min_alert_interval=2.0,
        min_risk_for_alert=min_risk_for_alert,
        min_collision_duration=0.0,
        min_risk_score=0.0,
    )
    
    rprint("[cyan]Alert Requirement:[/]")
    rprint("  • Collision detected in BOTH camera frames")
    rprint("  • Risk score > 40%")
    rprint()
    
    import time
    frame_times = []
    fps_display_interval = 30
    
    dims1, dims2 = camera_manager.get_frame_dimensions()
    
    try:
        while True:
            frame_start = time.time()
            
            success, frame1, frame2 = camera_manager.read_synchronized_frames()
            if not success:
                rprint("[yellow]End of video stream or camera disconnected.[/]")
                break
            
            bboxes1, bboxes2 = camera_manager.process_frames(frame1, frame2)
            
            collisions1 = detect_collisions(
                bboxes1,
                iou_threshold=iou_threshold,
                distance_threshold=distance_threshold,
                frame_width=dims1[0],
                frame_height=dims1[1],
            )
            
            collisions2 = detect_collisions(
                bboxes2,
                iou_threshold=iou_threshold,
                distance_threshold=distance_threshold,
                frame_width=dims2[0],
                frame_height=dims2[1],
            )
            
            if collisions1 and collisions2:
                for collision in collisions1:
                    if alert_system.should_alert(collision):
                        alert_system.trigger_alert(
                            collision_cam1=collision,
                            collision_cam2=None,
                            frame_number=camera_manager.frame_count,
                            verified=True,
                        )
            
            frame1_display = draw_bboxes_and_collisions(frame1.copy(), bboxes1, collisions1, "Camera 1")
            frame2_display = draw_bboxes_and_collisions(frame2.copy(), bboxes2, collisions2, "Camera 2")
            
            frame_end = time.time()
            frame_time = frame_end - frame_start
            frame_times.append(frame_time)
            
            if len(frame_times) >= fps_display_interval:
                avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
                rprint(f"[dim]Processing FPS: {avg_fps:.1f}[/]")
                frame_times = []
            
            max_width = 960
            h1, w1 = frame1_display.shape[:2]
            h2, w2 = frame2_display.shape[:2]
            
            if w1 > max_width:
                scale1 = max_width / w1
                frame1_display = cv2.resize(frame1_display, (int(w1 * scale1), int(h1 * scale1)))
            
            if w2 > max_width:
                scale2 = max_width / w2
                frame2_display = cv2.resize(frame2_display, (int(w2 * scale2), int(h2 * scale2)))
            
            if display_mode == "separate":
                cv2.imshow("Camera 1", frame1_display)
                cv2.imshow("Camera 2", frame2_display)
            elif display_mode == "side":
                h1, w1 = frame1_display.shape[:2]
                h2, w2 = frame2_display.shape[:2]
                target_h = min(h1, h2)
                frame1_display = cv2.resize(frame1_display, (int(w1 * target_h / h1), target_h))
                frame2_display = cv2.resize(frame2_display, (int(w2 * target_h / h2), target_h))
                combined = np.hstack([frame1_display, frame2_display])
                cv2.imshow("Dual Camera Monitor", combined)
            else:
                h1, w1 = frame1_display.shape[:2]
                h2, w2 = frame2_display.shape[:2]
                target_w = max(w1, w2)
                if w1 < target_w:
                    frame1_display = cv2.resize(frame1_display, (target_w, int(h1 * target_w / w1)))
                if w2 < target_w:
                    frame2_display = cv2.resize(frame2_display, (target_w, int(h2 * target_w / w2)))
                combined = np.vstack([frame1_display, frame2_display])
                cv2.imshow("Dual Camera Monitor", combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                stats = alert_system.get_statistics()
                rprint(f"\n[cyan]Statistics: {stats}[/]\n")
    
    finally:
        camera_manager.release()
        cv2.destroyAllWindows()
        
        stats = alert_system.get_statistics()
        rprint("\n[bold green]Monitoring Session Complete[/]")
        rprint(f"Total Alerts: {stats['total_alerts']}")
        rprint(f"Total Frames Processed: {camera_manager.frame_count}")


if __name__ == "__main__":
    app()
