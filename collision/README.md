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
   - `FACE_REG_*` entries drive the registration script (sample counts, confidence, capture delay, GPU flag, camera index).
   - `FACE_RECOG_*` entries control the streaming script (detection confidence, similarity threshold, detector size, camera index, GPU flag, optional video path/prompt).
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

- The stream window overlays bounding boxes and the predicted identity with similarity score.
- Press `q` to exit. Use `--threshold` to make recognition stricter or more permissive.

## Data Storage

Embeddings are written to `data/face_database.json`. You can safely delete this file to reset the registry.

Collision alerts are logged to `data/alerts/alerts_YYYY-MM-DD.json` for daily review.

## Collision Detection (Multi-Camera)

The system now supports **dual-camera collision detection** to monitor when people are too close or colliding in real-time.

### How It Works

1. **Dual Camera Verification**: Uses 2 cameras/videos from different angles
2. **Collision Detection**: Identifies when people's bounding boxes overlap or are too close
3. **Cross-Validation**: Only alerts when BOTH cameras detect the same collision (configurable)
4. **Risk Assessment**: Calculates risk levels (SAFE/LOW/MEDIUM/HIGH/CRITICAL) based on:
   - IoU (Intersection over Union) - how much bounding boxes overlap
   - Distance between people's centers
   - Frame size normalization

### Running Collision Monitoring

```powershell
# Using two webcams (camera 0 and camera 1)
python src/monitor_collision.py

# Using two video files
python src/monitor_collision.py --video1 "path\to\cam1.mp4" --video2 "path\to\cam2.mp4"

# Mix webcam and video
python src/monitor_collision.py --camera1 0 --video2 "path\to\cam2.mp4"

# Adjust sensitivity
python src/monitor_collision.py --iou-threshold 0.2 --distance-threshold 150

# Alert on any collision (don't require both cameras to confirm)
python src/monitor_collision.py --require-both-cameras false

# Only alert on high-risk collisions
python src/monitor_collision.py --min-risk-for-alert HIGH

# Enable audio alerts
python src/monitor_collision.py --enable-audio
```

### Configuration Options

Set defaults in `.env`:

```env
# Camera indices (0 = first webcam, 1 = second webcam)
COLLISION_CAMERA1_INDEX=0
COLLISION_CAMERA2_INDEX=1

# IoU threshold (0.1 = 10% overlap triggers detection)
COLLISION_IOU_THRESHOLD=0.1

# Distance threshold in pixels (200 = alert when centers < 200px apart)
COLLISION_DISTANCE_THRESHOLD=200.0

# Minimum risk level for alerts (SAFE/LOW/MEDIUM/HIGH/CRITICAL)
COLLISION_MIN_RISK_FOR_ALERT=MEDIUM

# Require both cameras to detect collision before alerting
COLLISION_REQUIRE_BOTH_CAMERAS=true

# Enable audio beeps for alerts
COLLISION_ENABLE_AUDIO=false
```

### Display Modes

Choose how to view the dual camera feeds:

```powershell
# Vertical stack (default)
python src/monitor_collision.py --display-mode stacked

# Horizontal side-by-side
python src/monitor_collision.py --display-mode side

# Two separate windows
python src/monitor_collision.py --display-mode separate
```

### Keyboard Controls

- Press `q` to quit
- Press `s` to show statistics summary

### Alert Logs

All collision alerts are saved to `data/alerts/alerts_YYYY-MM-DD.json` with:
- Timestamp
- People involved
- Risk level and score
- IoU values from both cameras
- Verification status
- Frame number

## Troubleshooting

- The first run downloads InsightFace models; keep an active internet connection.
- If detection is unreliable, increase lighting and adjust `--min-confidence`.
- For best mask performance, capture balanced samples with and without the mask and consider using different mask colors/forms during registration.
- If you see InsightFace fall back to CPU, double-check that `onnxruntime-gpu` is installed and that the NVIDIA driver exposes CUDA to ONNX Runtime.
