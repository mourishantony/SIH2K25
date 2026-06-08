"""Patch monitor_service.py: update _build_tracker to resolve YOLO model path absolutely."""
import sys
from pathlib import Path

monitor_file = Path("src/monitor_service.py")
if not monitor_file.exists():
    print("ERROR: src/monitor_service.py not found")
    sys.exit(1)

content = monitor_file.read_text(encoding="utf-8")

old_code = """\
    def _build_tracker(self) -> PersonTracker:
        use_gpu = recognition_settings.reid_embedder_gpu or self.use_gpu
        return PersonTracker(
            model_path=str(recognition_settings.reid_model_path or "yolov8n.pt"),
            detection_confidence=recognition_settings.reid_detector_conf,
            embedder_gpu=use_gpu,
            nms_iou=recognition_settings.reid_nms_iou,
            box_shrink=recognition_settings.reid_box_shrink,
            device='cuda' if use_gpu else None,
        )"""

new_code = """\
    def _build_tracker(self) -> PersonTracker:
        use_gpu = recognition_settings.reid_embedder_gpu or self.use_gpu

        # Resolve yolov8n.pt to absolute path so FastAPI finds it regardless
        # of which working directory uvicorn was launched from.
        raw_model_path = str(recognition_settings.reid_model_path or "yolov8n.pt")
        model_path = Path(raw_model_path)
        if not model_path.is_absolute():
            candidate_src  = ROOT_DIR / "src" / raw_model_path
            candidate_here = Path(__file__).resolve().parent / raw_model_path
            if candidate_src.exists():
                model_path = candidate_src
            elif candidate_here.exists():
                model_path = candidate_here

        print(f"[MonitorService] YOLO model path resolved -> {model_path}")

        return PersonTracker(
            model_path=str(model_path),
            detection_confidence=recognition_settings.reid_detector_conf,
            embedder_gpu=use_gpu,
            nms_iou=recognition_settings.reid_nms_iou,
            box_shrink=recognition_settings.reid_box_shrink,
            device='cuda' if use_gpu else None,
        )"""

if old_code in content:
    patched = content.replace(old_code, new_code)
    monitor_file.write_text(patched, encoding="utf-8")
    print("[FIX 2] Patched _build_tracker in monitor_service.py")
elif "candidate_src" in content:
    print("[FIX 2] _build_tracker already patched -- skipping")
else:
    print("[FIX 2] WARNING: Could not find exact old _build_tracker code to replace.")
    print("        The function may have a different format. Please patch manually.")
