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


def _capture_continuous(
    cap: cv2.VideoCapture,
    analyzer,
    target_count: int,
    min_confidence: float,
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

        status_line = f"Samples: {len(samples)}/{target_count}"
        capture_hint = "Click/Enter to capture | Put on/remove mask anytime | q to abort"

        if face is not None:
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, status_line, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 120), 2)
            cv2.putText(frame, capture_hint, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            if _capture_request:
                if time.time() - last_capture >= 0.05:
                    samples.append(face.normed_embedding.copy())
                    last_capture = time.time()
                    rprint(f"[cyan]Sample {len(samples)}/{target_count}[/] recorded")
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
            cv2.putText(frame, capture_hint, (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise typer.Abort()
        elif key == 13:  # Enter key
            _capture_request = True

    return samples


@app.command()
def register(
    name: str = typer.Argument(..., help="Person's name you want to register."),
    total_samples: int = typer.Option(50, help="Total number of samples to capture (with/without mask)."),
    camera_index: int = typer.Option(register_settings.camera_index, help="Webcam index passed to OpenCV."),
    min_confidence: float = typer.Option(register_settings.min_confidence, help="Minimum detector confidence to accept a face."),
    use_gpu: bool = typer.Option(register_settings.use_gpu, help="Use CUDAExecutionProvider when available (needs onnxruntime-gpu)."),
):
    """Register a new person in a single continuous session.

    Click the mouse or press Enter to capture each sample. You can put on or remove
    your mask at any time during the session. Press q to abort.
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
        rprint("[yellow]Click or press Enter to capture samples. Put on/remove mask anytime. Press q to abort.[/]")
        samples = _capture_continuous(
            shared_cam,
            analyzer,
            total_samples,
            min_confidence,
        )

    finally:
        if shared_cam is not None:
            shared_cam.release()
        cv2.destroyAllWindows()

    total = upsert_embeddings(name, samples)
    rprint(f"[bold green]{len(samples)} samples stored for {name}. Total entries: {total}[/]")


@app.command()
def unregister(
    name: str = typer.Argument(..., help="Name of the person to remove from the database."),
):
    """Remove a registered person from the face database."""
    from face_db import remove_person

    success = remove_person(name)
    if success:
        rprint(f"[bold green]{name} has been removed from the database.[/]")
    else:
        rprint(f"[bold red]{name} was not found in the database.[/]")


if __name__ == "__main__":
    app()
