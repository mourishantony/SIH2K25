import torch
import psutil
import time
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel


class GPUMonitor:
    
    def __init__(self):
        self.console = Console()
        self.start_time = time.time()
        self.frame_count = 0
        self.inference_times = []
        
    def get_gpu_stats(self) -> Dict[str, Any]:
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
        memory_cached = torch.cuda.memory_reserved(0) / 1024**2
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        
        gpu_util = (memory_allocated / memory_total) * 100
        
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "memory_allocated": memory_allocated,
            "memory_cached": memory_cached,
            "memory_total": memory_total,
            "memory_utilization": (memory_allocated / memory_total) * 100,
            "gpu_utilization": gpu_util,
            "cpu_percent": cpu_percent,
            "ram_percent": memory_info.percent,
            "cuda_version": torch.version.cuda,
        }
    
    def update_performance(self, inference_time: float):
        self.frame_count += 1
        self.inference_times.append(inference_time)
        
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        if not self.inference_times:
            return {"fps": 0, "avg_inference": 0, "total_time": 0}
        
        total_time = time.time() - self.start_time
        avg_inference = sum(self.inference_times) / len(self.inference_times)
        fps = self.frame_count / total_time if total_time > 0 else 0
        
        return {
            "fps": fps,
            "avg_inference_ms": avg_inference * 1000,
            "total_frames": self.frame_count,
            "total_time": total_time,
            "min_inference_ms": min(self.inference_times) * 1000,
            "max_inference_ms": max(self.inference_times) * 1000,
        }
    
    def display_stats(self):
        gpu_stats = self.get_gpu_stats()
        perf_stats = self.get_performance_stats()
        
        if "error" in gpu_stats:
            self.console.print(f"❌ [red]{gpu_stats['error']}[/red]")
            return
        
        table = Table(title="🚀 RTX 3050 Performance Monitor")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")
        
        table.add_row("GPU", gpu_stats["gpu_name"], "✅ Active")
        table.add_row("CUDA Version", gpu_stats["cuda_version"], "✅ Compatible")
        
        memory_status = "🟢 Good" if gpu_stats["memory_utilization"] < 80 else "🟡 High"
        table.add_row(
            "GPU Memory", 
            f"{gpu_stats['memory_allocated']:.0f}/{gpu_stats['memory_total']:.0f} MB",
            memory_status
        )
        table.add_row(
            "GPU Usage", 
            f"{gpu_stats['memory_utilization']:.1f}%",
            memory_status
        )
        
        fps_status = "🟢 Excellent" if perf_stats["fps"] > 25 else "🟡 Good" if perf_stats["fps"] > 15 else "🔴 Poor"
        table.add_row("FPS", f"{perf_stats['fps']:.1f}", fps_status)
        table.add_row("Avg Inference", f"{perf_stats['avg_inference_ms']:.1f} ms", "")
        table.add_row("Total Frames", f"{perf_stats['total_frames']}", "")
        
        cpu_status = "🟢 Good" if gpu_stats["cpu_percent"] < 80 else "🟡 High"
        table.add_row("CPU Usage", f"{gpu_stats['cpu_percent']:.1f}%", cpu_status)
        
        ram_status = "🟢 Good" if gpu_stats["ram_percent"] < 80 else "🟡 High"
        table.add_row("RAM Usage", f"{gpu_stats['ram_percent']:.1f}%", ram_status)
        
        self.console.clear()
        self.console.print(table)
    
    def create_live_display(self):
        def generate_display():
            while True:
                gpu_stats = self.get_gpu_stats()
                perf_stats = self.get_performance_stats()
                
                if "error" in gpu_stats:
                    yield Panel(f"❌ {gpu_stats['error']}", title="GPU Status")
                    continue
                
                content = f"""
🚀 [bold green]RTX 3050 GPU Status[/bold green]
├── Memory: {gpu_stats['memory_allocated']:.0f}/{gpu_stats['memory_total']:.0f} MB ({gpu_stats['memory_utilization']:.1f}%)
├── FPS: [bold]{perf_stats['fps']:.1f}[/bold]
├── Inference: {perf_stats['avg_inference_ms']:.1f} ms
├── Frames: {perf_stats['total_frames']}
└── CPU: {gpu_stats['cpu_percent']:.1f}% | RAM: {gpu_stats['ram_percent']:.1f}%
                """.strip()
                
                yield Panel(content, title="🎯 Collision Detection Monitor")
                time.sleep(1)
        
        return Live(generate_display(), refresh_per_second=1)


# Global monitor instance
gpu_monitor = GPUMonitor()


def start_gpu_monitoring():
    return gpu_monitor.create_live_display()


def update_gpu_stats(inference_time: float):
    gpu_monitor.update_performance(inference_time)


def display_gpu_summary():
    gpu_monitor.display_stats()


if __name__ == "__main__":
    monitor = GPUMonitor()
    
    import random
    for i in range(10):
        inference_time = random.uniform(0.015, 0.025)
        monitor.update_performance(inference_time)
        time.sleep(0.1)
    
    monitor.display_stats()