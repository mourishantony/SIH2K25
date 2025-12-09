from __future__ import annotations

import sys

from rich import print as rprint
from rich.console import Console
from rich.table import Table


def test_imports():
    rprint("\n[bold cyan]Testing Module Imports...[/]")
    
    modules = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("insightface", "insightface"),
        ("typer", "typer"),
        ("rich", "rich"),
        ("dotenv", "python-dotenv"),
    ]
    
    failed = []
    for module_name, package_name in modules:
        try:
            __import__(module_name)
            rprint(f"  ✓ {package_name}")
        except ImportError:
            rprint(f"  ✗ {package_name} [red](missing)[/]")
            failed.append(package_name)
    
    if failed:
        rprint(f"\n[red]Missing packages: {', '.join(failed)}[/]")
        rprint("[yellow]Install with: pip install " + " ".join(failed) + "[/]")
        return False
    
    rprint("[green]✓ All imports successful[/]")
    return True


def test_custom_modules():
    rprint("\n[bold cyan]Testing Custom Modules...[/]")
    
    try:
        from config import collision_settings, recognition_settings, register_settings
        rprint("  ✓ config.py")
    except Exception as e:
        rprint(f"  ✗ config.py [red]{e}[/]")
        return False
    
    try:
        from collision_detector import BoundingBox, Collision, detect_collisions
        rprint("  ✓ collision_detector.py")
    except Exception as e:
        rprint(f"  ✗ collision_detector.py [red]{e}[/]")
        return False
    
    try:
        from multi_camera import CameraStream, DualCameraManager
        rprint("  ✓ multi_camera.py")
    except Exception as e:
        rprint(f"  ✗ multi_camera.py [red]{e}[/]")
        return False
    
    try:
        from alert_system import AlertSystem, AlertEvent
        rprint("  ✓ alert_system.py")
    except Exception as e:
        rprint(f"  ✗ alert_system.py [red]{e}[/]")
        return False
    
    try:
        from face_db import load_facebank, save_facebank
        rprint("  ✓ face_db.py")
    except Exception as e:
        rprint(f"  ✗ face_db.py [red]{e}[/]")
        return False
    
    try:
        from vision import get_analyzer
        rprint("  ✓ vision.py")
    except Exception as e:
        rprint(f"  ✗ vision.py [red]{e}[/]")
        return False
    
    rprint("[green]✓ All custom modules loaded[/]")
    return True


def test_configuration():
    rprint("\n[bold cyan]Testing Configuration...[/]")
    
    try:
        from config import collision_settings
        
        table = Table(title="Collision Detection Settings")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Camera 1 Index", str(collision_settings.camera1_index))
        table.add_row("Camera 2 Index", str(collision_settings.camera2_index))
        table.add_row("IoU Threshold", str(collision_settings.iou_threshold))
        table.add_row("Distance Threshold", str(collision_settings.distance_threshold))
        table.add_row("Min Confidence", str(collision_settings.min_confidence))
        table.add_row("Use GPU", str(collision_settings.use_gpu))
        table.add_row("Enable Audio", str(collision_settings.enable_audio))
        table.add_row("Min Risk for Alert", collision_settings.min_risk_for_alert)
        table.add_row("Require Both Cameras", str(collision_settings.require_both_cameras))
        
        Console().print(table)
        rprint("[green]✓ Configuration loaded successfully[/]")
        return True
    except Exception as e:
        rprint(f"[red]✗ Configuration error: {e}[/]")
        return False


def test_face_database():
    rprint("\n[bold cyan]Testing Face Database...[/]")
    
    try:
        from face_db import load_facebank
        
        registry = load_facebank()
        if registry:
            rprint(f"[green]✓ Face database found with {len(registry)} people:[/]")
            for name, embeddings in registry.items():
                rprint(f"  - {name}: {len(embeddings)} samples")
        else:
            rprint("[yellow]⚠ Face database is empty[/]")
            rprint("[yellow]  Register people with: python src/register_face.py 'Name'[/]")
        
        return True
    except Exception as e:
        rprint(f"[red]✗ Database error: {e}[/]")
        return False


def test_collision_logic():
    rprint("\n[bold cyan]Testing Collision Detection Logic...[/]")
    
    try:
        from collision_detector import BoundingBox, calculate_iou, calculate_distance, detect_collisions
        
        bbox1 = BoundingBox(100, 100, 200, 200, "Person A", 0.95)
        bbox2 = BoundingBox(180, 180, 280, 280, "Person B", 0.92)
        bbox3 = BoundingBox(400, 400, 500, 500, "Person C", 0.88)
        
        iou = calculate_iou(bbox1, bbox2)
        rprint(f"  IoU (overlapping boxes): {iou:.3f}")
        
        dist = calculate_distance(bbox1, bbox2)
        rprint(f"  Distance (overlapping): {dist:.1f}px")
        
        dist_far = calculate_distance(bbox1, bbox3)
        rprint(f"  Distance (far apart): {dist_far:.1f}px")
        
        bboxes = [bbox1, bbox2, bbox3]
        collisions = detect_collisions(bboxes, iou_threshold=0.1, distance_threshold=200, frame_width=640, frame_height=480)
        
        rprint(f"\n  Detected {len(collisions)} collision(s):")
        for collision in collisions:
            rprint(f"    - {collision}")
        
        rprint("[green]✓ Collision detection logic working[/]")
        return True
    except Exception as e:
        rprint(f"[red]✗ Collision logic error: {e}[/]")
        import traceback
        traceback.print_exc()
        return False


def main():
    rprint("[bold green]═══════════════════════════════════════════════════[/]")
    rprint("[bold green]  Collision Detection System - Setup Verification[/]")
    rprint("[bold green]═══════════════════════════════════════════════════[/]")
    
    results = []
    
    results.append(("Package Imports", test_imports()))
    results.append(("Custom Modules", test_custom_modules()))
    results.append(("Configuration", test_configuration()))
    results.append(("Face Database", test_face_database()))
    results.append(("Collision Logic", test_collision_logic()))
    
    rprint("\n[bold cyan]═══════════════════════════════════════════════════[/]")
    rprint("[bold cyan]  Test Summary[/]")
    rprint("[bold cyan]═══════════════════════════════════════════════════[/]")
    
    for test_name, passed in results:
        status = "[green]✓ PASS[/]" if passed else "[red]✗ FAIL[/]"
        rprint(f"  {test_name:.<40} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        rprint("\n[bold green]✓ All tests passed! System ready to use.[/]")
        rprint("\n[cyan]Quick Start Commands:[/]")
        rprint("  1. Register a person: [yellow]python src/register_face.py 'Name'[/]")
        rprint("  2. Test recognition: [yellow]python src/recognize_face.py[/]")
        rprint("  3. Monitor collisions: [yellow]python src/monitor_collision.py[/]")
        return 0
    else:
        rprint("\n[bold red]✗ Some tests failed. Please fix the issues above.[/]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
