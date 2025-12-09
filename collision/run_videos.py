import typer
from pathlib import Path
from rich import print as rprint
from rich.console import Console
from rich.table import Table
import os

app = typer.Typer()
console = Console()

def list_available_videos():
    video_dir = Path("src")
    video_files = list(video_dir.glob("*.mp4"))
    
    if not video_files:
        rprint("[red]No video files found in src/ directory![/red]")
        return []
    
    table = Table(title="🎬 Available Video Files")
    table.add_column("Index", style="cyan", width=6)
    table.add_column("Filename", style="green")
    table.add_column("Size", style="yellow")
    
    for idx, video in enumerate(video_files, 1):
        size = video.stat().st_size / (1024*1024)
        table.add_row(str(idx), video.name, f"{size:.1f} MB")
    
    console.print(table)
    return video_files

@app.command()
def run(
    video1: str = typer.Option("src/vid4.mp4", help="Path to first video file"),
    video2: str = typer.Option("src/vid4_1.mp4", help="Path to second video file"),
    display_mode: str = typer.Option("side", help="Display mode: side, stacked, separate"),
    iou_threshold: float = typer.Option(0.1, help="IoU threshold for collision detection"),
    distance_threshold: float = typer.Option(200, help="Distance threshold for proximity"),
    use_gpu: bool = typer.Option(True, help="Use RTX 3050 GPU acceleration"),
):
    rprint("\n🚀 [bold green]RTX 3050 GPU Video Collision Detection[/bold green]")
    rprint("=" * 50)
    
    if not Path(video1).exists():
        rprint(f"[red]❌ Video 1 not found: {video1}[/red]")
        return
    
    if not Path(video2).exists():
        rprint(f"[red]❌ Video 2 not found: {video2}[/red]")
        return
    
    rprint(f"📹 [cyan]Video 1: {video1}[/cyan]")
    rprint(f"📹 [cyan]Video 2: {video2}[/cyan]")
    rprint(f"🖥️  [yellow]Display: {display_mode}[/yellow]")
    rprint(f"⚙️  [blue]GPU: {'✅ ENABLED' if use_gpu else '❌ DISABLED'}[/blue]")
    rprint()
    
    from src.monitor_collision import app as monitor_app
    import sys
    
    sys.argv = [
        "monitor_collision.py",
        "monitor",
        "--video1", video1,
        "--video2", video2,
        "--display-mode", display_mode,
        "--iou-threshold", str(iou_threshold),
        "--distance-threshold", str(distance_threshold),
    ]
    
    if use_gpu:
        sys.argv.append("--use-gpu")
    else:
        sys.argv.append("--no-use-gpu")
    
    monitor_app()

@app.command()
def list_videos():
    rprint("\n🎬 [bold]Available Video Files[/bold]")
    video_files = list_available_videos()
    
    if video_files:
        rprint(f"\n📁 Found {len(video_files)} video files in src/ directory")
        rprint("\n💡 [cyan]Usage examples:[/cyan]")
        rprint("  python run_videos.py run --video1 src/vid1.mp4 --video2 src/vid2.mp4")
        rprint("  python run_videos.py run --video1 src/vid3.mp4 --video2 src/vid3_1.mp4")

@app.command()
def quick_test(
    video_pair: int = typer.Option(1, help="Video pair: 1=vid4+vid4_1, 2=vid3+vid3_1")
):
    video_pairs = {
        1: ("src/vid4.mp4", "src/vid4_1.mp4"),
        2: ("src/vid3.mp4", "src/vid3_1.mp4"),
    }
    
    if video_pair not in video_pairs:
        rprint(f"[red]Invalid video pair {video_pair}. Available: 1, 2[/red]")
        return
    
    video1, video2 = video_pairs[video_pair]
    
    rprint(f"\n🎯 [bold]Quick Test - Video Pair {video_pair}[/bold]")
    run(video1=video1, video2=video2, display_mode="side", use_gpu=True)

@app.command() 
def performance_test():
    rprint("\n⚡ [bold]GPU Performance Test with Videos[/bold]")
    
    os.system("python test_gpu_performance.py")
    
    rprint("\n🎬 [cyan]Testing with actual video files...[/cyan]")
    os.system("python run_videos.py run --video1 src/vid4.mp4 --video2 src/vid4_1.mp4")

if __name__ == "__main__":
    app()