# Patient Contact Tracing System

This workspace provides a dual-camera contact tracing system for monitoring patient interactions. It uses InsightFace (ArcFace) for face recognition with mask support, YOLO + DeepSORT + OSNet for person tracking, and collision detection for risk assessment.

## Prerequisites

1. Activate the provided virtual environment or your own Python 3.10+ environment.
2. Install the dependencies:
   ```powershell
   C:/Users/mourish/Desktop/sih_01/venv/Scripts/python.exe -m pip install -r requirements.txt
   ```
3. Ensure your cameras (laptop cam + external cam) are connected and accessible by OpenCV.
4. Optional (GPU acceleration): uninstall the CPU runtime and install the CUDA build instead.
   ```powershell
   C:/Users/mourish/Desktop/sih_01/venv/Scripts/python.exe -m pip uninstall -y onnxruntime
   C:/Users/mourish/Desktop/sih_01/venv/Scripts/python.exe -m pip install onnxruntime-gpu
   ```

## Configuration via .env

1. Copy `.env.example` to `.env`.
2. Edit the values to set your preferred defaults:
   - `FACE_REG_*` entries control registration (samples, confidence, camera index, GPU flag)
   - `FACE_REID_*` entries control person tracking (YOLO confidence, NMS thresholds, box shrink)
   - `FRONT_*` / `SIDE_*` entries set the dual-camera sources
   - `CONTACT_*` entries configure risk calculation parameters
   - `COLLISION_*` entries set collision detection and alert thresholds
3. CLI flags override environment variables when provided.

Example settings:

```env
FACE_REG_TOTAL_SAMPLES=50
FACE_REG_USE_GPU=true
COLLISION_MIN_RISK_FOR_ALERT=0.4
FRONT_CAMERA_INDEX=0
SIDE_CAMERA_INDEX=1
```

## Core Workflow

This project provides three main commands for patient contact tracing:

### 1. Register a Person

Capture face samples with and without a mask using the live camera.

```powershell
C:/Users/mourish/Desktop/sih_01/venv/Scripts/python.exe src/register_face.py register "Person Name"
```

- The script runs a single continuous session collecting 50 samples by default
- **Click the video preview or press `Enter`** to capture each sample manually
- The person can put on or remove their mask at any time during the session—there's no forced phase separation
- Press `q` in the preview window to abort

Set `FACE_REG_USE_GPU=true` in `.env` for GPU acceleration. Adjust `--total-samples`, `--camera-index`, and `--min-confidence` as needed.

### 2. Unregister a Person

Remove a registered identity from the database:

```powershell
C:/Users/mourish/Desktop/sih_01/venv/Scripts/python.exe src/register_face.py unregister "Person Name"
```

This deletes all embeddings for that person from `data/face_database.json`.

### 3. Monitor Contacts (Dual-Camera Contact Tracing)

Run the dual-camera monitoring system with integrated face recognition, person tracking, and collision detection:

```powershell
C:/Users/mourish/Desktop/sih_01/venv/Scripts/python.exe src/monitor_contacts.py
```

The monitor pairs a front view camera (laptop) with a side view camera (external) or two video files to confirm close contacts when *both* perspectives show overlapping person boxes.

#### How It Works

**Face Recognition (InsightFace + ArcFace):**
- Recognizes registered patients even when wearing masks
- Uses 512-dimensional face embeddings for robust matching
- Similarity threshold controlled via environment settings

**Person Re-ID + Tracking (YOLO + DeepSORT + OSNet):**
- YOLO detects person bounding boxes in each frame
- DeepSORT tracks individuals across frames using OSNet embeddings
- Maintains identity even when face is occluded or turned away
- Orange fallback boxes drawn when tracker merges nearby people
- Configure via `FACE_REID_DET_CONF`, `FACE_REID_NMS_IOU`, `FACE_REID_BOX_SHRINK`

**Risk Calculation:**

Each confirmed contact pair accumulates risk using:
```
R_new = R_old + (BaseRate × MaskModifier × Δt) + EventPenalty
```

- `CONTACT_BASE_RATE`: Risk accumulation per second (default 0.02)
- `CONTACT_EVENT_PENALTY`: One-time penalty when contact starts (default 0.05)
- `CONTACT_MASK_EFFECT`: Reduces risk when mask is detected (0.5 = 50% reduction)
- Mask detection uses a trained classifier on the Kaggle face-mask-detection dataset

**Collision Detection + Alerts:**
- Measures IoU (Intersection over Union) + center-to-center distance for each pair
- Blends metrics into normalized risk score (0.0 = SAFE → 1.0 = CRITICAL)
- `CollisionTracker` accumulates duration and frame counts per pair
- `AlertSystem` triggers alerts when:
  - Contact duration ≥ `COLLISION_ALERT_DURATION` (default 10s)
  - Risk score ≥ `COLLISION_MIN_RISK_FOR_ALERT` (default 0.4)
  - Both cameras confirm contact (if `COLLISION_REQUIRE_BOTH_CAMERAS=true`)
- Logs alerts to `data/alerts/alerts_YYYY-MM-DD.json`
- Optional audio beep via `COLLISION_ENABLE_AUDIO=true`
- Cooldown period prevents duplicate alerts: `COLLISION_ALERT_COOLDOWN` (default 12s)

**Contact Logging:**
- Each registered person gets a folder: `Contact_Details/<Name>/contacts.json`
- Format: `{person_name: {timestamps: [...], risk_percent: 0-100}}`
- Bidirectional logging: both parties record the same incident
- Timestamps mark when contact started (ISO 8601 format)
- Risk percentage shows cumulative exposure (capped at 100%)

**Configuration:**
- `FRONT_CAMERA_INDEX` / `SIDE_CAMERA_INDEX`: Camera device indices (default 0, 1)
- `FRONT_VIDEO_PROMPT` / `SIDE_VIDEO_PROMPT`: Show file picker for video input
- `CONTACT_SYNC_WINDOW`: Max timing difference between views (default 0.5s)
- `CONTACT_OVERLAP_THRESHOLD`: Minimum IoU to trigger contact detection (default 0.12)
- Press `q` to stop monitoring

The preview window shows both camera feeds side-by-side with bounding boxes, identity labels, and real-time risk overlays.

## Data Storage

- **Face embeddings:** `data/face_database.json` (delete to reset registry)
- **Contact logs:** `Contact_Details/<PersonName>/contacts.json` (delete individual folders to clear history)
- **Alert logs:** `data/alerts/alerts_YYYY-MM-DD.json` (one file per day)

## Troubleshooting

- The first run downloads InsightFace models; keep an active internet connection.
- If detection is unreliable, increase lighting and adjust `FACE_REG_MIN_CONFIDENCE`.
- For best mask performance, capture balanced samples with and without the mask during registration.
- If you see InsightFace fall back to CPU, double-check that `onnxruntime-gpu` is installed and that the NVIDIA driver exposes CUDA to ONNX Runtime.
- If bounding boxes disappear when people stand close, the system uses face-to-body fallback boxes (orange) to maintain identity tracking.
- Adjust `COLLISION_IOU_THRESHOLD` and `COLLISION_DISTANCE_THRESHOLD` to balance sensitivity vs false positives.

## System Flow Summary

1. **Register patients** using `register_face.py register "Name"` with 50 samples (mask optional)
2. **Monitor contacts** using `monitor_contacts.py`:
   - InsightFace recognizes faces (even with masks)
   - YOLO + DeepSORT tracks full body movement
   - Collision detector measures proximity using IoU + distance
   - Risk accumulates over time with mask modifiers
   - Alerts trigger when thresholds exceeded
   - Contact logs record timestamps + risk percentages
3. **Review data** in `Contact_Details/` and `data/alerts/`
4. **Unregister patients** when needed using `register_face.py unregister "Name"`

## Technology Stack

- **Face Recognition:** InsightFace (ArcFace), ONNX Runtime
- **Person Detection:** YOLO v8
- **Person Tracking:** DeepSORT + OSNet
- **Mask Classification:** Custom CNN (trained on Kaggle dataset)
- **Collision Detection:** IoU + Euclidean distance with risk scoring
- **Alert System:** Duration-gated with cooldown periods
- **UI:** OpenCV + Rich console output
