# Face Recognition with Mask Support

This workspace contains a minimal end-to-end workflow to register new identities with a webcam and recognize them in real time, even when they wear a mask. It relies on InsightFace (ArcFace embeddings) for robust facial representations.

## Prerequisites

1. Activate the provided virtual environment or your own Python 3.10+ environment.
2. Install the dependencies:
   ```powershell
   C:/Users/mourish/Desktop/sih_01/venv/Scripts/python.exe -m pip install -r requirements.txt
   ```
3. Ensure your webcam (laptop cam counts) is connected and accessible by OpenCV.
4. Optional (GPU acceleration): uninstall the CPU runtime and install the CUDA build instead.
   ```powershell
   C:/Users/mourish/Desktop/sih_01/venv/Scripts/python.exe -m pip uninstall -y onnxruntime
   C:/Users/mourish/Desktop/sih_01/venv/Scripts/python.exe -m pip install onnxruntime-gpu
   ```

## Configuration via .env

1. Copy `.env.example` to `.env`.
2. Edit the values to set your preferred defaults:
   - `FACE_REG_*` entries drive the registration script (sample counts, confidence, capture delay, GPU flag, camera index, optional video prompt).
   - `FACE_RECOG_*` entries control the streaming script (detection confidence, similarity threshold, detector size, camera index, GPU flag, optional video path/prompt, Re-ID options).
   - `FACE_RECOG_VIDEO_PATH` lets you default the recognition input to a video file instead of the webcam when set to an absolute or workspace-relative path.
3. CLI flags still override the environment, so you can temporarily tweak a value without editing `.env`.

Example tweaks:

```env
FACE_RECOG_THRESHOLD=0.40
FACE_RECOG_USE_GPU=true
FACE_REG_CAPTURE_DELAY=0.35
```

## Registering a Person

Capture both unmasked and masked samples so the encoder learns to recognize the person in either state.

```powershell
C:/Users/mourish/Desktop/sih_01/venv/Scripts/python.exe src/register_face.py "Person Name"
```

Set `FACE_REG_USE_GPU=true` in `.env` if you want registration to default to CUDA without passing flags every run.

If you prefer registering from pre-recorded clips instead of the webcam, set `FACE_REG_VIDEO_PROMPT=true`. The script will show a file chooser for each phase (mask off/on). Close the dialog or press cancel to fall back to the live camera for that phase.

- Phase 1 runs with the mask **off**. Keep your face centered; samples are taken automatically every ~0.45s.
- After Phase 1, the script pauses. Put on your mask and press `Enter` in the terminal to resume Phase 2.
- Press `q` in the preview window at any time to abort.

You can tweak options such as `--unmasked-samples`, `--masked-samples`, `--camera-index`, and `--capture-delay` if needed.

## Real-Time Recognition

```powershell
C:/Users/mourish/Desktop/sih_01/venv/Scripts/python.exe src/recognize_face.py
```

`FACE_RECOG_USE_GPU=true` keeps GPU acceleration enabled with no CLI flags.

To change the default similarity guard, set `FACE_RECOG_THRESHOLD` inside `.env` (CLI `--threshold` still wins when supplied).

To analyze an MP4/AVI instead of the live camera, pass `--video-path path/to/file.mp4` (or set `FACE_RECOG_VIDEO_PATH` in `.env`). When the flag/variable is present, `--camera-index` is ignored and frames are pulled from the file until it ends.

When `FACE_RECOG_VIDEO_PROMPT=true` (or `--video-prompt`), the app pops up a file chooser so you can select the clip interactively; close/cancel the dialog to keep streaming the webcam.

### Person Re-ID + Tracking (DeepSORT + OSNet)

1. Install PyTorch with CUDA suited to your GPU if you have not already. The provided `requirements.txt` references the generic CPU wheels; for GPU builds follow the [official PyTorch install selector](https://pytorch.org/get-started/locally/) and then re-run `pip install -r requirements.txt` to pick up the remaining packages.
2. Download the YOLOv8 weights you prefer (default `yolov8n.pt` downloads automatically from Ultralytics on first use). If you have a local custom model, set `FACE_REID_DETECTOR=/path/to/model.pt`.
3. Flip `FACE_RECOG_ENABLE_REID=true` in `.env`. Optional knobs: `FACE_REID_DET_CONF` to tighten or loosen the person detector confidence, `FACE_REID_DETECTOR` to point at a different YOLO checkpoint, and `FACE_REID_EMBEDDER_GPU` to let the OSNet embedder run on the GPU (defaults to CPU if false).

Workflow: the camera first recognizes faces as before. When a face is confidently labeled, its bounding box center is matched to the closest tracked person (using DeepSORT with OSNet embeddings). From that point onward, the tracker keeps a bounding box around the entire body, even if the face turns away or becomes occluded. Once a track leaves the scene it is removed automatically, and the next positive face match reassigns the identity.

- The stream window overlays bounding boxes and the predicted identity with similarity score.
- Press `q` to exit. Use `--threshold` to make recognition stricter or more permissive.

## Data Storage

Embeddings are written to `data/face_database.json`. You can safely delete this file to reset the registry.

## Troubleshooting

- The first run downloads InsightFace models; keep an active internet connection.
- If detection is unreliable, increase lighting and adjust `--min-confidence`.
- For best mask performance, capture balanced samples with and without the mask and consider using different mask colors/forms during registration.
- If you see InsightFace fall back to CPU, double-check that `onnxruntime-gpu` is installed and that the NVIDIA driver exposes CUDA to ONNX Runtime.
