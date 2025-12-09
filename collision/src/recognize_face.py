from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import typer
from rich import print as rprint

from config import recognition_settings
from face_db import flatten_registry, load_facebank
from ui_utils import pick_video_file
from vision import get_analyzer

_video_path_default: Optional[Path] = (
    Path(recognition_settings.video_path) if recognition_settings.video_path else None
)


def _predict_identity(
    embedding: np.ndarray,
    names: np.ndarray,
    embeddings: np.ndarray,
    threshold: float,
) -> tuple[str, float]:
    if embeddings.size == 0:
        return "Unknown", 0.0
    similarities = embeddings @ embedding
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])
    if best_score < threshold:
        return "Unknown", best_score
    return str(names[best_idx]), best_score


def run_recognition(
    camera_index: int = typer.Option(recognition_settings.camera_index, help="Webcam index passed to OpenCV."),
    min_confidence: float = typer.Option(recognition_settings.min_confidence, help="Minimum detector confidence to accept a face."),
    threshold: float = typer.Option(recognition_settings.threshold, help="Cosine similarity threshold to accept a prediction."),
    det_size: int = typer.Option(recognition_settings.det_size, help="Square detection size fed into InsightFace."),
    use_gpu: bool = typer.Option(recognition_settings.use_gpu, help="Use CUDAExecutionProvider when available (needs onnxruntime-gpu)."),
    video_path: Optional[Path] = typer.Option(
        _video_path_default,
        help="Optional path to an MP4/AVI/etc. If set, frames are read from the file instead of the live camera.",
    ),
    video_prompt: bool = typer.Option(
        recognition_settings.video_prompt,
        "--video-prompt/--no-video-prompt",
        help="Prompt for a video path at runtime; leave blank to stream the webcam.",
    ),
):

    registry = load_facebank()
    if not registry:
        raise typer.Exit("No embeddings found. Run register_face.py first.")
    names, embeddings = flatten_registry(registry)
    embeddings = embeddings.astype(np.float32)

    analyzer = get_analyzer((det_size, det_size), use_gpu=use_gpu)

    if video_prompt:
        picked = pick_video_file()
        video_path = Path(picked) if picked else None

    if video_path is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise typer.BadParameter(f"Cannot open video file: {video_path}")
        source_label = f"file {video_path}"
    else:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise typer.BadParameter("Cannot open the requested webcam.")
        source_label = f"camera {camera_index}"

    rprint(f"[bold green]Streaming from {source_label}. Press q to exit.[/]")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if video_path is not None:
                    rprint("[yellow]Reached end of video file.[/]")
                    break
                raise RuntimeError("Unable to read frame from the webcam.")

            faces = analyzer.get(frame)
            for face in faces:
                if face.det_score < min_confidence:
                    continue
                x1, y1, x2, y2 = map(int, face.bbox)
                embedding = face.normed_embedding.astype(np.float32)
                name, score = _predict_identity(embedding, names, embeddings, threshold)

                color = (0, 200, 100) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{name} {score:.2f}"
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    typer.run(run_recognition)
