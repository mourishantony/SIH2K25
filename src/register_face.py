from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np
import typer
from rich import print as rprint

from config import register_settings
from face_db import upsert_embeddings
from ui_utils import pick_video_file
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
    source_label: str,
    is_video_source: bool,
) -> List[np.ndarray]:
    samples: List[np.ndarray] = []
    last_capture = 0.0
    while len(samples) < target_count:
        ret, frame = cap.read()
        if not ret:
            if is_video_source:
                raise typer.BadParameter(
                    f"{source_label} ended before capturing enough samples. Provide a longer clip."
                )
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
    video_prompt: bool = typer.Option(
        register_settings.video_prompt,
        "--video-prompt/--no-video-prompt",
        help="When enabled, select phase videos through a file chooser instead of the live webcam.",
    ),
):
    """Register a new person by capturing embeddings with and without a mask."""

    rprint("[bold green]Face registration starting...[/]")

    shared_cam: Optional[cv2.VideoCapture] = None
    if not video_prompt:
        shared_cam = cv2.VideoCapture(camera_index)
        if not shared_cam.isOpened():
            raise typer.BadParameter("Cannot open the requested webcam.")

    analyzer = get_analyzer(use_gpu=use_gpu)

    def _ensure_webcam() -> cv2.VideoCapture:
        nonlocal shared_cam
        if shared_cam is None:
            shared_cam = cv2.VideoCapture(camera_index)
            if not shared_cam.isOpened():
                raise typer.BadParameter("Cannot open the requested webcam.")
        return shared_cam

    def _select_capture(phase_name: str):
        if video_prompt:
            picked = pick_video_file(f"Select video for {phase_name}")
            if picked:
                cap = cv2.VideoCapture(picked)
                if not cap.isOpened():
                    raise typer.BadParameter(f"Cannot open video file: {picked}")
                rprint(f"[cyan]{phase_name}[/] using video: {picked}")
                return cap, f"video {picked}", True, True
            rprint("[yellow]No video selected; falling back to webcam for this phase.[/]")
        cam = _ensure_webcam()
        return cam, f"camera {camera_index}", False, False

    try:
        if video_prompt:
            rprint("[yellow]Phase 1:[/] choose a clip with mask OFF when the dialog appears.")
        else:
            rprint("[yellow]Phase 1:[/] keep your mask OFF and face the camera. Press q to abort.")
        cap_phase, label, is_video_source, should_release = _select_capture("Phase 1 (no mask)")
        try:
            unmasked = _capture_phase(
                cap_phase,
                analyzer,
                unmasked_samples,
                "No mask",
                min_confidence,
                capture_delay,
                label,
                is_video_source,
            )
        finally:
            if should_release:
                cap_phase.release()

        masked: List[np.ndarray] = []
        if masked_samples > 0:
            if video_prompt:
                rprint("[yellow]Phase 2:[/] choose a clip with mask ON when prompted.")
            else:
                typer.secho("Put on your mask, then press ENTER to continue...", fg=typer.colors.YELLOW)
                typer.prompt("Ready? hit ENTER when the mask is on", default="", show_default=False)
                rprint("[yellow]Phase 2:[/] keep your mask ON and face the camera.")
            cap_phase, label, is_video_source, should_release = _select_capture("Phase 2 (mask on)")
            try:
                masked = _capture_phase(
                    cap_phase,
                    analyzer,
                    masked_samples,
                    "Mask on",
                    min_confidence,
                    capture_delay,
                    label,
                    is_video_source,
                )
            finally:
                if should_release:
                    cap_phase.release()

    finally:
        if shared_cam is not None:
            shared_cam.release()
        cv2.destroyAllWindows()

    stored = unmasked + masked
    total = upsert_embeddings(name, stored)
    rprint(f"[bold green]{len(stored)} samples stored for {name}. Total entries: {total}[/]")


if __name__ == "__main__":
    app()
