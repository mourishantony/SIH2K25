from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import typer
from rich import print as rprint

from config import recognition_settings
from face_db import flatten_registry, load_facebank
from reid_tracker import PersonTracker, TrackInfo
from ui_utils import pick_video_file
from vision import get_analyzer

_video_path_default: Optional[Path] = (
    Path(recognition_settings.video_path) if recognition_settings.video_path else None
)

BBox = Tuple[int, int, int, int]
WINDOW_NAME = "Recognition"
_window_size: Tuple[int, int] = (0, 0)


def _sync_window_to_frame(frame: np.ndarray) -> None:
    """Resize the preview window so it matches the incoming frame size."""
    global _window_size
    height, width = frame.shape[:2]
    if (width, height) != _window_size:
        cv2.resizeWindow(WINDOW_NAME, width, height)
        _window_size = (width, height)


def _match_face_to_track(face_bbox: BBox, tracks: Sequence[TrackInfo]) -> Optional[TrackInfo]:
    if not tracks:
        return None
    fx1, fy1, fx2, fy2 = face_bbox
    cx = (fx1 + fx2) / 2
    cy = (fy1 + fy2) / 2
    best_track: Optional[TrackInfo] = None
    best_area = float("inf")
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            area = (x2 - x1) * (y2 - y1)
            if area < best_area:
                best_area = area
                best_track = track
    return best_track


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
    enable_reid: bool = typer.Option(recognition_settings.enable_reid, help="Enable body tracking with DeepSORT + OSNet."),
    reid_det_conf: float = typer.Option(
        recognition_settings.reid_detector_conf,
        help="Minimum detector confidence for YOLO person boxes when ReID is enabled.",
    ),
    reid_detector: Path = typer.Option(
        Path(recognition_settings.reid_model_path or "yolov8n.pt"),
        exists=False,
        help="YOLO checkpoint used for person detection when ReID is enabled.",
    ),
):
    """Start real-time recognition with webcam feed."""

    registry = load_facebank()
    if not registry:
        raise typer.Exit("No embeddings found. Run register_face.py first.")
    names, embeddings = flatten_registry(registry)
    embeddings = embeddings.astype(np.float32)

    analyzer = get_analyzer((det_size, det_size), use_gpu=use_gpu)
    person_tracker: Optional[PersonTracker] = None
    track_identities: Dict[int, str] = {}
    identity_claims: Dict[str, int] = {}
    if enable_reid:
        person_tracker = PersonTracker(
            model_path=str(reid_detector),
            detection_confidence=reid_det_conf,
            embedder_gpu=recognition_settings.reid_embedder_gpu or use_gpu,
        )

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

    global _window_size

    rprint(f"[bold green]Streaming from {source_label}. Press q to exit.[/]")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    _window_size = (0, 0)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if video_path is not None:
                    rprint("[yellow]Reached end of video file.[/]")
                    break
                raise RuntimeError("Unable to read frame from the webcam.")

            active_tracks: Sequence[TrackInfo] = ()
            if person_tracker is not None:
                active_tracks = person_tracker.update(frame)

            _sync_window_to_frame(frame)

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

                if name != "Unknown" and person_tracker is not None:
                    matched = _match_face_to_track((x1, y1, x2, y2), active_tracks)
                    if matched is not None:
                        previous_track = identity_claims.get(name)
                        if previous_track is not None and previous_track != matched.track_id:
                            track_identities.pop(previous_track, None)
                        track_identities[matched.track_id] = name
                        identity_claims[name] = matched.track_id

            if person_tracker is not None:
                active_ids = {track.track_id for track in active_tracks}
                track_identities = {tid: label for tid, label in track_identities.items() if tid in active_ids}
                identity_claims = {label: tid for label, tid in identity_claims.items() if tid in active_ids}

                for track in active_tracks:
                    x1, y1, x2, y2 = track.bbox
                    assigned = track_identities.get(track.track_id, f"Track {track.track_id}")
                    color = (255, 191, 0) if assigned.startswith("Track") else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        assigned,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    typer.run(run_recognition)
