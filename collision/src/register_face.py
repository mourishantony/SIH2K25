from __future__ import annotations

import time
from typing import List

import cv2
import numpy as np
import typer
from rich import print as rprint

from config import register_settings
from face_db import upsert_embeddings
from vision import get_analyzer

app = typer.Typer(add_completion=False)


def _select_primary_face(faces, min_confidence: float):
    candidates = [face for face in faces if face.det_score >= min_confidence]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
    )


def _capture_phase(
    cap: cv2.VideoCapture,
    analyzer,
    target_count: int,
    phase_label: str,
    min_confidence: float,
    capture_delay: float,
) -> List[np.ndarray]:
    samples: List[np.ndarray] = []
    last_capture = 0.0
    while len(samples) < target_count:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Unable to read frame from the webcam.")

        faces = analyzer.get(frame)
        face = _select_primary_face(faces, min_confidence)

        if face is not None:
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{phase_label}: {len(samples)+1}/{target_count}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 120), 2)
            if time.time() - last_capture >= capture_delay:
                samples.append(face.normed_embedding.copy())
                last_capture = time.time()
                rprint(f"[cyan]{phase_label}[/] sample {len(samples)}/{target_count} recorded")
        else:
            cv2.putText(
                frame,
                "Align your face with the camera...",
                (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Registration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise typer.Abort()

    return samples


@app.command()
def register(
    name: str = typer.Argument(..., help="Person's name you want to register."),
    unmasked_samples: int = typer.Option(register_settings.unmasked_samples, help="How many unmasked samples to capture."),
    masked_samples: int = typer.Option(register_settings.masked_samples, help="How many masked samples to capture after the first phase."),
    camera_index: int = typer.Option(register_settings.camera_index, help="Webcam index passed to OpenCV."),
    min_confidence: float = typer.Option(register_settings.min_confidence, help="Minimum detector confidence to accept a face."),
    capture_delay: float = typer.Option(register_settings.capture_delay, help="Seconds between stored samples to avoid near duplicates."),
    use_gpu: bool = typer.Option(register_settings.use_gpu, help="Use CUDAExecutionProvider when available (needs onnxruntime-gpu)."),
):

    rprint("[bold green]Face registration starting...[/]")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise typer.BadParameter("Cannot open the requested webcam.")

    analyzer = get_analyzer(use_gpu=use_gpu)

    try:
        rprint("[yellow]Phase 1:[/] keep your mask OFF and face the camera. Press q to abort.")
        unmasked = _capture_phase(cap, analyzer, unmasked_samples, "No mask", min_confidence, capture_delay)

        masked: List[np.ndarray] = []
        if masked_samples > 0:
            typer.secho("Put on your mask, then press ENTER to continue...", fg=typer.colors.YELLOW)
            typer.prompt("Ready? hit ENTER when the mask is on", default="", show_default=False)
            rprint("[yellow]Phase 2:[/] keep your mask ON and face the camera.")
            masked = _capture_phase(cap, analyzer, masked_samples, "Mask on", min_confidence, capture_delay)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    stored = unmasked + masked
    total = upsert_embeddings(name, stored)
    rprint(f"[bold green]{len(stored)} samples stored for {name}. Total entries: {total}[/]")


if __name__ == "__main__":
    app()
