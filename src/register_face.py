from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import typer
from rich import print as rprint

from config import register_settings
from face_db import upsert_embeddings
from vision import get_analyzer

WINDOW_NAME = "Registration"
_window_size: Tuple[int, int] = (0, 0)
_capture_request: bool = False


def _sync_window_to_frame(frame: np.ndarray) -> None:
    """Resize the preview window so it matches the incoming frame size."""
    global _window_size
    height, width = frame.shape[:2]
    if (width, height) != _window_size:
        cv2.resizeWindow(WINDOW_NAME, width, height)
        _window_size = (width, height)


def _mouse_callback(event, x, y, flags, param):  # noqa: D401
    """Request a single capture on left mouse click."""
    global _capture_request
    if event == cv2.EVENT_LBUTTONDOWN:
        _capture_request = True

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
    global _capture_request
    samples: List[np.ndarray] = []
    last_capture = 0.0
    while len(samples) < target_count:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Unable to read frame from the webcam.")

        _sync_window_to_frame(frame)

        faces = analyzer.get(frame)
        face = _select_primary_face(faces, min_confidence)

        status_line = f"{phase_label}: {len(samples)}/{target_count}"
        capture_hint = "Click to capture"

        if face is not None:
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, status_line, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 120), 2)
            cv2.putText(frame, capture_hint, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if _capture_request:
                # Debounce rapid clicks a bit to avoid accidental double-captures
                if time.time() - last_capture >= 0.05:
                    samples.append(face.normed_embedding.copy())
                    last_capture = time.time()
                    rprint(f"[cyan]{phase_label}[/] sample {len(samples)}/{target_count} recorded")
                _capture_request = False
        else:
            cv2.putText(
                frame,
                "No face detected - align with camera.",
                (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(frame, capture_hint, (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)
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
    """Register a new person using ONLY the webcam.

    Click the left mouse button once to start capturing; click again to pause.
    Press q at any time to abort.
    """

    global _window_size

    rprint("[bold green]Face registration starting...[/]")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    _window_size = (0, 0)

    shared_cam: Optional[cv2.VideoCapture] = cv2.VideoCapture(camera_index)
    if not shared_cam.isOpened():
        raise typer.BadParameter("Cannot open the requested webcam.")

    analyzer = get_analyzer(use_gpu=use_gpu)

    cv2.setMouseCallback(WINDOW_NAME, _mouse_callback)

    try:
        rprint("[yellow]Phase 1:[/] keep your mask OFF and face the camera. Click to start, click again to pause. Press q to abort.")
        unmasked = _capture_phase(
            shared_cam,
            analyzer,
            unmasked_samples,
            "No mask",
            min_confidence,
            capture_delay,
        )

        masked: List[np.ndarray] = []
        if masked_samples > 0:
            typer.secho("Put on your mask, then press ENTER to continue...", fg=typer.colors.YELLOW)
            typer.prompt("Ready? hit ENTER when the mask is on", default="", show_default=False)
            rprint("[yellow]Phase 2:[/] keep your mask ON. Click to start/pause capture. Press q to abort.")
            masked = _capture_phase(
                shared_cam,
                analyzer,
                masked_samples,
                "Mask on",
                min_confidence,
                capture_delay,
            )

    finally:
        if shared_cam is not None:
            shared_cam.release()
        cv2.destroyAllWindows()

    stored = unmasked + masked
    total = upsert_embeddings(name, stored)
    rprint(f"[bold green]{len(stored)} samples stored for {name}. Total entries: {total}[/]")


if __name__ == "__main__":
    app()
