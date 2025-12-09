import torch
import cv2
import numpy as np
import time
from rich import print as rprint
from rich.console import Console
from rich.table import Table

def test_gpu_setup():
    console = Console()
    
    rprint("\n🚀 [bold green]RTX 3050 GPU Performance Test[/bold green]")
    rprint("=" * 50)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        rprint(f"✅ [green]CUDA Available: {torch.cuda.is_available()}[/green]")
        rprint(f"🔥 [cyan]GPU: {gpu_name}[/cyan]")
        rprint(f"💾 [yellow]Memory: {gpu_memory:.1f} GB[/yellow]")
        rprint(f"🔧 [blue]CUDA Version: {torch.version.cuda}[/blue]")
        
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
        memory_cached = torch.cuda.memory_reserved(0) / 1024**2
        rprint(f"📊 Memory Allocated: {memory_allocated:.1f} MB")
        rprint(f"📊 Memory Cached: {memory_cached:.1f} MB")
        
    else:
        rprint("❌ [red]CUDA not available![/red]")
        return False
    
    rprint("\n🧪 [bold]Testing GPU Performance...[/bold]")
    
    device = torch.device("cuda")
    size = 1000
    
    start_time = time.time()
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    
    start_time = time.time()
    a_gpu = torch.randn(size, size, device=device)
    b_gpu = torch.randn(size, size, device=device)
    c_gpu = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    
    table = Table(title="Performance Comparison")
    table.add_column("Device", style="cyan")
    table.add_column("Time (seconds)", style="magenta")
    table.add_column("Speedup", style="green")
    
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    table.add_row("CPU", f"{cpu_time:.4f}", "-")
    table.add_row("RTX 3050", f"{gpu_time:.4f}", f"{speedup:.1f}x faster")
    
    console.print(table)
    return True


def test_yolov8_gpu():
    try:
        from ultralytics import YOLO
        rprint("\n🎯 [bold]Testing YOLOv8 GPU Performance...[/bold]")
        
        model = YOLO('yolov8n.pt')
        model.to('cuda')
        
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        for _ in range(3):
            _ = model.predict(test_image, device='cuda', verbose=False)
        
        num_runs = 10
        start_time = time.time()
        
        for _ in range(num_runs):
            results = model.predict(test_image, device='cuda', verbose=False)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_runs
        fps = 1 / avg_time
        
        rprint(f"🚀 [green]YOLOv8 GPU Performance:[/green]")
        rprint(f"   Average inference time: {avg_time*1000:.1f} ms")
        rprint(f"   Estimated FPS: {fps:.1f}")
        
        return True
        
    except ImportError:
        rprint("❌ [red]Ultralytics not installed![/red]")
        return False
    except Exception as e:
        rprint(f"❌ [red]YOLOv8 test failed: {e}[/red]")
        return False


def test_opencv_gpu():
    rprint("\n📹 [bold]Testing OpenCV GPU Support...[/bold]")
    
    build_info = cv2.getBuildInformation()
    has_cuda = 'CUDA:' in build_info and 'YES' in build_info.split('CUDA:')[1].split('\n')[0]
    
    if has_cuda:
        rprint("✅ [green]OpenCV built with CUDA support[/green]")
    else:
        rprint("⚠️ [yellow]OpenCV without CUDA (using CPU)[/yellow]")
    
    test_image = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
    
    start_time = time.time()
    blurred_cpu = cv2.GaussianBlur(test_image, (15, 15), 0)
    cpu_time = time.time() - start_time
    
    rprint(f"🖥️  CPU processing time: {cpu_time*1000:.1f} ms")
    
    return True


if __name__ == "__main__":
    gpu_ok = test_gpu_setup()
    
    if gpu_ok:
        test_yolov8_gpu()
        test_opencv_gpu()
        
        rprint("\n🎉 [bold green]GPU Setup Complete for RTX 3050![/bold green]")
        rprint("💡 [cyan]Ready for high-performance collision detection![/cyan]")
    else:
        rprint("\n❌ [bold red]GPU Setup Failed![/bold red]")
        rprint("🔧 [yellow]Please check CUDA installation[/yellow]")