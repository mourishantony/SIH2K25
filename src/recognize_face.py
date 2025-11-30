from __future__ import annotations

import cv2
import numpy as np
import typer
from rich import print as rprint

from config import recognition_settings
from face_db import flatten_registry, load_facebank
from vision import get_analyzer

app = typer.Typer(add_completion=False)


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


@app.command()
def watch(
    camera_index: int = typer.Option(recognition_settings.camera_index, help="Webcam index passed to OpenCV."),
    min_confidence: float = typer.Option(recognition_settings.min_confidence, help="Minimum detector confidence to accept a face."),
    threshold: float = typer.Option(recognition_settings.threshold, help="Cosine similarity threshold to accept a prediction."),
    det_size: int = typer.Option(recognition_settings.det_size, help="Square detection size fed into InsightFace."),
    use_gpu: bool = typer.Option(recognition_settings.use_gpu, help="Use CUDAExecutionProvider when available (needs onnxruntime-gpu)."),
):
    """Start real-time recognition with webcam feed."""

    registry = load_facebank()
    if not registry:
        raise typer.Exit("No embeddings found. Run register_face.py first.")
    names, embeddings = flatten_registry(registry)
    embeddings = embeddings.astype(np.float32)

    analyzer = get_analyzer((det_size, det_size), use_gpu=use_gpu)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise typer.BadParameter("Cannot open the requested webcam.")

    rprint("[bold green]Streaming... Press q to exit.[/]")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
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
    app()
