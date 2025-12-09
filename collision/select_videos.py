import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from run_video_collision import main
from rich import print as rprint

def select_videos():
    available_videos = [
        ("src/vid1.mp4", "src/vid2.mp4", "Videos 1 & 2"),
        ("src/vid3.mp4", "src/vid3_1.mp4", "Videos 3 & 3_1 (Matching pair)"),
        ("src/vid4.mp4", "src/vid4_1.mp4", "Videos 4 & 4_1 (Matching pair)"),  
        ("src/vid5.mp4", "src/vid6.mp4", "Videos 5 & 6"),
    ]
    
    rprint("\n🎯 [bold green]RTX 3050 GPU Collision Detection - Video Selection[/bold green]")
    rprint("="*60)
    rprint("Available video combinations:")
    
    for i, (vid1, vid2, desc) in enumerate(available_videos, 1):
        rprint(f"  {i}. {desc}")
        rprint(f"     📹 {vid1} + {vid2}")
    
    rprint("\n🔥 [yellow]All videos will be processed with GPU acceleration![/yellow]")
    rprint("⚡ Expected performance: ~25-30 FPS with RTX 3050")
    rprint("🎮 Controls: Press 'q' to quit, 's' for statistics")
    
    while True:
        try:
            choice = input(f"\nSelect videos (1-{len(available_videos)}) or press Enter for default (4): ").strip()
            
            if choice == "":
                choice = 4
            else:
                choice = int(choice)
                
            if 1 <= choice <= len(available_videos):
                return available_videos[choice-1]
            else:
                rprint(f"[red]Please enter a number between 1 and {len(available_videos)}[/red]")
                
        except ValueError:
            rprint("[red]Please enter a valid number[/red]")
        except KeyboardInterrupt:
            rprint("\n[yellow]Cancelled by user[/yellow]")
            return None

if __name__ == "__main__":
    selection = select_videos()
    
    if selection:
        video1, video2, desc = selection
        rprint(f"\n🚀 [bold green]Starting collision detection with {desc}[/bold green]")
        
        import run_video_collision
        run_video_collision.video1_path = video1
        run_video_collision.video2_path = video2
        
        def updated_main():
            from run_video_collision import CollisionDualCameraManager, AlertSystem, detect_collisions
            from run_video_collision import draw_bboxes_and_collisions, rprint, cv2, np, time
            
            rprint("[🚀 bold green]Starting RTX 3050 GPU-Accelerated Collision Detection[🚀 /bold green]")
            rprint(f"Video 1: {video1}")
            rprint(f"Video 2: {video2}")
            rprint("[bold cyan]COLLISION DETECTION ONLY - No Face Recognition[/bold cyan]")
            rprint("[yellow]Controls: Press 'q' to quit, 's' for statistics[/yellow]")
            rprint()
            
            try:
                camera_manager = CollisionDualCameraManager(
                    source1=video1,
                    source2=video2,
                    use_gpu=True,
                    min_confidence=0.3,
                )
                
                alert_system = AlertSystem(
                    enable_audio=False,
                    enable_logging=True,
                    min_alert_interval=2.0,
                    min_risk_for_alert="MEDIUM",
                    min_collision_duration=0.0,
                    min_risk_score=0.0,
                )
                
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
                    
                    collisions1 = detect_collisions(
                        bboxes1,
                        iou_threshold=0.1,
                        distance_threshold=200,
                        frame_width=dims1[0],
                        frame_height=dims1[1],
                    )
                    
                    collisions2 = detect_collisions(
                        bboxes2,
                        iou_threshold=0.1,
                        distance_threshold=200,
                        frame_width=dims2[0],
                        frame_height=dims2[1],
                    )
                    
                    if collisions1 and collisions2:
                        for collision in collisions1:
                            if alert_system.should_alert(collision):
                                rprint(f"🚨 [red]COLLISION ALERT[/red]: {collision}")
                                alert_system.trigger_alert(
                                    collision_cam1=collision,
                                    collision_cam2=None,
                                    frame_number=camera_manager.frame_count,
                                    verified=True,
                                )
                    
                    frame1_display = draw_bboxes_and_collisions(frame1.copy(), bboxes1, collisions1, "Video 1")
                    frame2_display = draw_bboxes_and_collisions(frame2.copy(), bboxes2, collisions2, "Video 2")
                    
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
                    cv2.imshow(f"RTX 3050 GPU Collision Detection - {desc}", combined)
                    
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
        
        updated_main()