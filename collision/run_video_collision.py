import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import cv2
import numpy as np
import time
from rich import print as rprint

from alert_system import AlertSystem
from collision_detector import detect_collisions, CollisionTracker
from collision_multi_camera import CollisionDualCameraManager

def draw_bboxes_and_collisions(frame, bboxes, collisions, camera_name):
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


def draw_enhanced_collisions(frame, all_collisions, significant_collisions, camera_name):
    significant_ids = {id(c.bbox1) for c in significant_collisions} | {id(c.bbox2) for c in significant_collisions}
    all_collision_ids = {id(c.bbox1) for c in all_collisions} | {id(c.bbox2) for c in all_collisions}
    
    for collision in all_collisions:
        for bbox in [collision.bbox1, collision.bbox2]:
            bbox_id = id(bbox)
            
            if bbox_id in significant_ids:
                color = (0, 165, 255)
                thickness = 4
            elif bbox_id in all_collision_ids:
                color = (0, 255, 255)
                thickness = 3
            else:
                color = (0, 255, 0)
                thickness = 2
            
            cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, thickness)
            
            label = f"{bbox.person_name} ({bbox.confidence:.2f})"
            cv2.putText(frame, label, (bbox.x1, bbox.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    y_offset = 60
    for collision in all_collisions:
        is_significant = collision in significant_collisions
        
        if is_significant:
            if collision.risk_score >= 0.7:
                color = (128, 0, 255)
            elif collision.risk_score >= 0.5:
                color = (0, 0, 255)
            else:
                color = (0, 165, 255)
        else:
            color = (0, 255, 255)
        
        center1 = collision.bbox1.center
        center2 = collision.bbox2.center
        cv2.line(frame, (int(center1[0]), int(center1[1])), (int(center2[0]), int(center2[1])), color, 3)
        
        mid_x = int((center1[0] + center2[0]) / 2)
        mid_y = int((center1[1] + center2[1]) / 2)
        
        if is_significant:
            warning = f"⚠ {collision.risk_level} ⚠"
            details = f"Risk: {collision.risk_score:.1%} | {collision.duration:.1f}s"
        else:
            remaining = 10.0 - collision.duration
            warning = f"⏱ {remaining:.1f}s to alert"
            details = f"Risk: {collision.risk_score:.1%} | Duration: {collision.duration:.1f}s"
        
        cv2.putText(frame, warning, (mid_x - 80, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, details, (mid_x - 80, mid_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    info_text = f"{camera_name} | All: {len(all_collisions)}, Significant (10s+): {len(significant_collisions)}"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    legend_y = frame.shape[0] - 80
    cv2.putText(frame, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Orange: 10s+ contact", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    cv2.putText(frame, "Yellow: Developing", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, "Green: Safe", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame


def main():
    video1_path = "src/vid5.mp4"
    video2_path = "src/vid6.mp4"
    
    rprint("[🚀 bold green]Starting RTX 3050 GPU-Accelerated Collision Detection[🚀 /bold green]")
    rprint(f"Video 1: {video1_path}")
    rprint(f"Video 2: {video2_path}")
    rprint("[bold cyan]COLLISION DETECTION ONLY - No Face Recognition[/bold cyan]")
    rprint("[yellow]Controls: Press 'q' to quit, 's' for statistics[/yellow]")
    rprint()
    
    try:
        camera_manager = CollisionDualCameraManager(
            source1=video1_path,
            source2=video2_path,
            use_gpu=True,
            min_confidence=0.3,
        )
        
        alert_system = AlertSystem(
            enable_audio=False,
            enable_logging=True,
            min_alert_interval=2.0,
            min_risk_for_alert="MEDIUM",
            min_collision_duration=10.0,
            min_risk_score=0.40,
            high_risk_threshold=0.40,
            enable_high_risk_json=True,
        )
        
        tracker1 = CollisionTracker(time_threshold=10.0)
        tracker2 = CollisionTracker(time_threshold=10.0)
        
        dims1, dims2 = camera_manager.get_frame_dimensions()
        
        frame_times = []
        fps_display_interval = 30
        
        rprint("[green]Starting video processing...[/green]")
        
        while True:
            frame_start = time.time()
            
            success, frame1, frame2 = camera_manager.read_synchronized_frames()
            if not success:
                rprint("[yellow]End of video files[/yellow]")
                break
            
            bboxes1, bboxes2 = camera_manager.process_frames(frame1, frame2)
            
            raw_collisions1 = detect_collisions(
                bboxes1,
                iou_threshold=0.1,
                distance_threshold=200,
                frame_width=dims1[0],
                frame_height=dims1[1],
            )
            
            raw_collisions2 = detect_collisions(
                bboxes2,
                iou_threshold=0.1,
                distance_threshold=200,
                frame_width=dims2[0],
                frame_height=dims2[1],
            )
            
            collisions1 = tracker1.update_collisions(raw_collisions1)
            collisions2 = tracker2.update_collisions(raw_collisions2)
            
            significant_collisions1 = tracker1.get_significant_collisions(collisions1)
            significant_collisions2 = tracker2.get_significant_collisions(collisions2)
            
            for collision in collisions1:
                alert_system.save_high_risk_collision_json(collision, camera_manager.frame_count)
            for collision in collisions2:
                alert_system.save_high_risk_collision_json(collision, camera_manager.frame_count)
            
            if significant_collisions1 and significant_collisions2:
                for collision in significant_collisions1:
                    if alert_system.should_alert(collision):
                        rprint(f"🚨 [red]COLLISION ALERT[/red] (10s+ contact): {collision}")
                        alert_system.trigger_alert(
                            collision_cam1=collision,
                            collision_cam2=None,
                            frame_number=camera_manager.frame_count,
                            verified=True,
                        )
            
            frame1_display = draw_enhanced_collisions(frame1.copy(), collisions1, significant_collisions1, "Video 1")
            frame2_display = draw_enhanced_collisions(frame2.copy(), collisions2, significant_collisions2, "Video 2")
            
            frame_end = time.time()
            frame_time = frame_end - frame_start
            frame_times.append(frame_time)
            
            if len(frame_times) >= fps_display_interval:
                avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
                rprint(f"[dim]GPU Processing FPS: {avg_fps:.1f}[/]")
                frame_times = []
            
            max_width = 800
            h1, w1 = frame1_display.shape[:2]
            h2, w2 = frame2_display.shape[:2]
            
            if w1 > max_width:
                scale1 = max_width / w1
                frame1_display = cv2.resize(frame1_display, (int(w1 * scale1), int(h1 * scale1)))
            
            if w2 > max_width:
                scale2 = max_width / w2
                frame2_display = cv2.resize(frame2_display, (int(w2 * scale2), int(h2 * scale2)))
            
            h1, w1 = frame1_display.shape[:2]
            h2, w2 = frame2_display.shape[:2]
            target_h = min(h1, h2)
            
            frame1_display = cv2.resize(frame1_display, (int(w1 * target_h / h1), target_h))
            frame2_display = cv2.resize(frame2_display, (int(w2 * target_h / h2), target_h))
            
            combined = np.hstack([frame1_display, frame2_display])
            cv2.imshow("RTX 3050 GPU Collision Detection - Dual Videos", combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                stats = alert_system.get_statistics()
                rprint(f"\n[cyan]Statistics: {stats}[/]\n")
    
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        
    finally:
        if 'camera_manager' in locals():
            camera_manager.release()
        cv2.destroyAllWindows()
        
        if 'alert_system' in locals():
            stats = alert_system.get_statistics()
            rprint("\n[bold green]Collision Detection Session Complete[/]")
            rprint(f"Total Alerts: {stats.get('total_alerts', 0)}")
            if 'camera_manager' in locals():
                rprint(f"Total Frames Processed: {camera_manager.frame_count}")

if __name__ == "__main__":
    main()