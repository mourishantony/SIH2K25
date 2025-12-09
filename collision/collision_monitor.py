from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import cv2
import numpy as np
import typer
from rich import print as rprint

from alert_system import AlertSystem
from collision_detector import detect_collisions, verify_collision_across_cameras
from collision_multi_camera import CollisionDualCameraManager

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
    
    return frame


@app.command()
def monitor(
    camera1: int = typer.Option(0, help="First camera index (or use --video1 for video file)"),
    camera2: int = typer.Option(1, help="Second camera index (or use --video2 for video file)"),
    video1: Optional[Path] = typer.Option(None, help="Path to first video file (overrides --camera1)"),
    video2: Optional[Path] = typer.Option(None, help="Path to second video file (overrides --camera2)"),
    iou_threshold: float = typer.Option(0.1, help="IoU threshold for collision detection"),
    distance_threshold: float = typer.Option(200, help="Distance threshold (pixels) for proximity detection"),
    min_confidence: float = typer.Option(0.3, help="Minimum detection confidence"),
    use_gpu: bool = typer.Option(True, help="Use RTX 3050 GPU acceleration (recommended)"),
    enable_audio: bool = typer.Option(False, help="Enable audio alerts for collisions"),
    min_risk_for_alert: str = typer.Option("MEDIUM", help="Minimum risk level for alerts (SAFE/LOW/MEDIUM/HIGH/CRITICAL)"),
    require_both_cameras: bool = typer.Option(True, help="Only alert when collision detected in BOTH cameras"),
    display_mode: str = typer.Option("side", help="Display mode: 'side' (horizontal, default), 'stacked' (vertical), or 'separate' (two windows)"),
    headless: bool = typer.Option(False, help="Run without GUI display (fixes OpenCV window errors)"),
    show_distance_m: bool = typer.Option(False, help="Overlay approximate real-world distance between people (requires calibration fx)"),
):
    source1 = str(video1) if video1 else camera1
    source2 = str(video2) if video2 else camera2
    
    rprint("[🚀 bold green]Starting RTX 3050 GPU-Accelerated Collision Monitoring[🚀 /bold green]")
    rprint(f"Video/Camera 1: {source1}")
    rprint(f"Video/Camera 2: {source2}")
    rprint(f"IoU Threshold: {iou_threshold}")
    rprint(f"Distance Threshold: {distance_threshold}px")
    gpu_status = "[green]ENABLED[/green]" if use_gpu else "[yellow]DISABLED[/yellow]"
    rprint(f"GPU Acceleration: {gpu_status}")
    rprint(f"Require Both Cameras: {require_both_cameras}")
    display_status = "[yellow]HEADLESS MODE[/yellow]" if headless else "[green]GUI ENABLED[/green]"
    rprint(f"Display Mode: {display_status}")
    if show_distance_m:
        rprint("[cyan]Distance overlay enabled (meters, approximate)[/]")
    rprint("[bold cyan]COLLISION DETECTION ONLY - No Face Recognition[/bold cyan]")
    rprint()
    
    try:
        camera_manager = CollisionDualCameraManager(
            source1=source1,
            source2=source2,
            use_gpu=use_gpu,
            min_confidence=min_confidence,
        )
    except Exception as e:
        rprint(f"[red]Error initializing cameras: {e}[/red]")
        return
    
    alert_system = AlertSystem(
        enable_audio=enable_audio,
        enable_logging=True,
        min_alert_interval=3.0,
        min_risk_for_alert=min_risk_for_alert,
        min_collision_duration=1.0,
        min_risk_score=0.25,
        high_risk_threshold=0.40,
    )
    
    rprint("[cyan]Alert Configuration:[/]")
    rprint("  • Pure collision detection (no face recognition)")
    rprint("  • GPU-accelerated person detection")
    rprint("  • Real-time risk assessment")
    rprint()
    
    import time
    frame_times = []
    fps_display_interval = 30
    
    collision_start_times = {}
    collision_frame_counts = {}
    last_alert_times = {}
    status_shown = {}
    
    dims1, dims2 = camera_manager.get_frame_dimensions()

    fx = None
    if show_distance_m:
        try:
            from src.distance_utils import load_fx_from_calibration
            fx = load_fx_from_calibration()
            rprint(f"[cyan]Loaded calibration fx={fx:.2f}[/]")
        except Exception as e:
            rprint(f"[yellow]Distance overlay disabled: {e}[/yellow]")
            rprint("[yellow]Run calibration: python src\\calibration.py --images \"data\\calib\\*.jpg\" --out \"data\\calibration_cam.json\"[/yellow]")
            show_distance_m = False
    
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
            
            current_verified_collisions = []
            
            if collisions1 and collisions2:
                for c1 in collisions1:
                    for c2 in collisions2:
                        people1 = set([c1.person1, c1.person2])
                        people2 = set([c2.person1, c2.person2])
                        
                        if people1 == people2:
                            max_risk = max(c1.risk_score, c2.risk_score)
                            collision_id = f"{min(c1.person1, c1.person2)}_{max(c1.person1, c1.person2)}"
                            
                            current_time = time.time()
                            if collision_id not in collision_start_times:
                                collision_start_times[collision_id] = current_time
                                collision_frame_counts[collision_id] = 0
                            
                            duration = current_time - collision_start_times[collision_id]
                            collision_frame_counts[collision_id] += 1
                            
                            verified_collision = c1 if c1.risk_score >= c2.risk_score else c2
                            verified_collision.risk_score = max_risk
                            verified_collision.duration = duration
                            verified_collision.frame_count = collision_frame_counts[collision_id]
                            
                            current_verified_collisions.append((verified_collision, c1, c2, collision_id))
                            break
            
            active_collision_ids = {collision_id for _, _, _, collision_id in current_verified_collisions}
            ended_collisions = set(collision_start_times.keys()) - active_collision_ids
            for collision_id in ended_collisions:
                del collision_start_times[collision_id]
                del collision_frame_counts[collision_id]
            
            for verified_collision, c1, c2, collision_id in current_verified_collisions:
                if verified_collision.duration >= 2.0 and verified_collision.risk_score >= 0.35:
                    if collision_id not in last_alert_times or (time.time() - last_alert_times[collision_id]) >= 5.0:
                        last_alert_times[collision_id] = time.time()
                        
                        rprint(f"[red]❗ SUSTAINED COLLISION ALERT[/red]")
                        rprint(f"[red]   People: {verified_collision.person1} ↔ {verified_collision.person2}[/red]")
                        rprint(f"[red]   Duration: {verified_collision.duration:.1f}s | Risk: {verified_collision.risk_score:.1%}[/red]")
                        rprint(f"[red]   Frame Count: {verified_collision.frame_count} | Status: Both Videos[/red]")
                        
                        alert_system.trigger_alert(
                            collision_cam1=c1,
                            collision_cam2=c2,
                            frame_number=camera_manager.frame_count,
                            verified=True,
                        )
                        
                        try:
                            alert_system.save_high_risk_collision_json(verified_collision, camera_manager.frame_count)
                            rprint(f"[cyan]💾 Significant collision saved to JSON[/cyan]")
                        except Exception as e:
                            rprint(f"[yellow]JSON save error: {e}[/yellow]")
                        rprint()
                    
                elif verified_collision.duration >= 1.0:
                    if collision_id not in status_shown or (time.time() - status_shown[collision_id]) >= 3.0:
                        status_shown[collision_id] = time.time()
                        rprint(f"[yellow]🔍 Monitoring: {verified_collision.person1} ↔ {verified_collision.person2} ({verified_collision.duration:.1f}s, {verified_collision.risk_score:.1%})[/yellow]")
            
            frame1_display = draw_bboxes_and_collisions(frame1.copy(), bboxes1, collisions1, "Video 1")
            frame2_display = draw_bboxes_and_collisions(frame2.copy(), bboxes2, collisions2, "Video 2")

            if show_distance_m and fx is not None:
                try:
                    from src.distance_utils import real_world_distance_meters
                    for c in collisions1:
                        d_m = real_world_distance_meters(
                            (c.bbox1.x1, c.bbox1.y1, c.bbox1.x2, c.bbox1.y2),
                            (c.bbox2.x1, c.bbox2.y1, c.bbox2.x2, c.bbox2.y2),
                            fx,
                            1.70,
                        )
                        mid_x = int((c.bbox1.center[0] + c.bbox2.center[0]) / 2)
                        mid_y = int((c.bbox1.center[1] + c.bbox2.center[1]) / 2)
                        cv2.putText(
                            frame1_display,
                            f"~{d_m:.2f} m",
                            (mid_x + 10, mid_y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                    for c in collisions2:
                        d_m = real_world_distance_meters(
                            (c.bbox1.x1, c.bbox1.y1, c.bbox1.x2, c.bbox1.y2),
                            (c.bbox2.x1, c.bbox2.y1, c.bbox2.x2, c.bbox2.y2),
                            fx,
                            1.70,
                        )
                        mid_x = int((c.bbox1.center[0] + c.bbox2.center[0]) / 2)
                        mid_y = int((c.bbox1.center[1] + c.bbox2.center[1]) / 2)
                        cv2.putText(
                            frame2_display,
                            f"~{d_m:.2f} m",
                            (mid_x + 10, mid_y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                except Exception as e:
                    rprint(f"[yellow]Distance overlay error: {e}[/yellow]")
            
            frame_end = time.time()
            frame_time = frame_end - frame_start
            frame_times.append(frame_time)
            
            if len(frame_times) >= fps_display_interval:
                avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
                rprint(f"[dim]Processing FPS: {avg_fps:.1f} | GPU Accelerated[/]")
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
            
            if not headless:
                try:
                    if display_mode == "separate":
                        cv2.imshow("Video 1", frame1_display)
                        cv2.imshow("Video 2", frame2_display)
                    elif display_mode == "side":
                        h1, w1 = frame1_display.shape[:2]
                        h2, w2 = frame2_display.shape[:2]
                        target_h = min(h1, h2)
                        frame1_display = cv2.resize(frame1_display, (int(w1 * target_h / h1), target_h))
                        frame2_display = cv2.resize(frame2_display, (int(w2 * target_h / h2), target_h))
                        combined = np.hstack([frame1_display, frame2_display])
                        cv2.imshow("Dual Video Collision Monitor", combined)
                    else:
                        h1, w1 = frame1_display.shape[:2]
                        h2, w2 = frame2_display.shape[:2]
                        target_w = max(w1, w2)
                        if w1 < target_w:
                            frame1_display = cv2.resize(frame1_display, (target_w, int(h1 * target_w / w1)))
                        if w2 < target_w:
                            frame2_display = cv2.resize(frame2_display, (target_w, int(h2 * target_w / w2)))
                        combined = np.vstack([frame1_display, frame2_display])
                        cv2.imshow("Dual Video Collision Monitor", combined)
                    
                    key = cv2.waitKey(1) & 0xFF
                    
                except Exception as display_error:
                    rprint(f"[yellow]Display error: {display_error}[/yellow]")
                    rprint("[cyan]Saving frames as images instead...[/cyan]")
                    
                    import os
                    os.makedirs("collision_frames", exist_ok=True)
                    
                    if display_mode == "side":
                        h1, w1 = frame1_display.shape[:2]
                        h2, w2 = frame2_display.shape[:2]
                        target_h = min(h1, h2)
                        frame1_resized = cv2.resize(frame1_display, (int(w1 * target_h / h1), target_h))
                        frame2_resized = cv2.resize(frame2_display, (int(w2 * target_h / h2), target_h))
                        combined = np.hstack([frame1_resized, frame2_resized])
                        cv2.imwrite(f"collision_frames/frame_{camera_manager.frame_count:06d}.jpg", combined)
                    else:
                        cv2.imwrite(f"collision_frames/cam1_frame_{camera_manager.frame_count:06d}.jpg", frame1_display)
                        cv2.imwrite(f"collision_frames/cam2_frame_{camera_manager.frame_count:06d}.jpg", frame2_display)
                    
                    if camera_manager.frame_count % 30 == 0:
                        rprint(f"[dim]Saved frame {camera_manager.frame_count} to collision_frames/[/dim]")
                    
                    headless = True
                    key = 0xFF
            else:
                import time
                time.sleep(0.01)
                key = 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                stats = alert_system.get_statistics()
                rprint(f"\n[cyan]Statistics: {stats}[/]\n")
    
    finally:
        camera_manager.release()
        if not headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        
        stats = alert_system.get_statistics()
        rprint("\n[bold green]Collision Monitoring Session Complete[/]")
        rprint(f"Total Alerts: {stats['total_alerts']}")
        rprint(f"Total Frames Processed: {camera_manager.frame_count}")


if __name__ == "__main__":
    app()