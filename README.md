# Patient Contact Tracing System

A comprehensive dual-camera contact tracing system with a web interface for hospital MDR (Multi-Drug Resistant) patient monitoring. Built for **SIH 2025 Grand Finale**.

## Features

- **Face Recognition**: InsightFace (ArcFace) with mask support
- **Person Tracking**: YOLO + DeepSORT + OSNet for real-time tracking
- **Collision Detection**: Monitors close contact between individuals
- **MDR Management**: Track and manage multi-drug resistant patients
- **Contact Alerts**: Email and web notifications for MDR contacts
- **Web Dashboard**: React-based admin interface with role-based access
- **AI Monitoring**: Integrated video/webcam processing through web interface

## Tech Stack

- **Frontend**: React 18, Vite, TailwindCSS
- **Backend**: FastAPI, Python 3.10+
- **Database**: MongoDB
- **AI/ML**: InsightFace, YOLO, DeepSORT, OSNet
- **Real-time**: WebSocket for live frame streaming

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB (running locally or remote)
- Webcam (for face registration and monitoring)

### 2. Backend Setup

```powershell
# Install Python dependencies
pip install -r requirements.txt

# Configure environment (edit .env file)
# Set MONGODB_URI, JWT_SECRET_KEY, SMTP settings, etc.

# Start the backend server
python start_server.py
```

Backend will be available at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

### 3. Frontend Setup

```powershell
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will be available at: `http://localhost:3000`

### 4. Default Admin Login

On first run, a default admin user is created:
- **Username**: `admin`
- **Password**: `admin123`

⚠️ **Change the password after first login!**

---

## User Roles & Permissions

The system has three user roles with different access levels:

| Role | Accessible Pages |
|------|-----------------|
| **Admin** | Dashboard, Register Person, Registered Persons, MDR Management, Alerts, Unknown Persons, AI Monitoring, User Management |
| **EHR User** | Dashboard, Registered Persons, MDR Management, Alerts |
| **Officer** | Dashboard, Register Person, Registered Persons, Unknown Persons, AI Monitoring |

Only administrators can create, edit, or delete user accounts.

---

## Web Application Pages

1. **Login**: User authentication (admin-managed accounts only)
2. **Dashboard**: System overview with stats and recent activity
3. **Register Person**: Capture face images via webcam (50 samples)
4. **Registered Persons**: View, edit, delete registered individuals
5. **MDR Management**: Mark/unmark patients as MDR, view contacts
6. **Alerts**: MDR contact notifications with snapshots
7. **Unknown Persons**: Track unidentified individuals detected by the system
8. **AI Monitoring**: Web-based video/webcam processing with live preview
9. **User Management**: Admin-only page for managing system users

---

## AI Monitoring (Web Interface)

The monitoring system is now integrated into the web application:

### Video File Mode
1. Navigate to **AI Monitoring** page
2. Select **Video File** mode
3. Upload front and side camera videos
4. Select both videos and click **Start Monitoring**
5. View live processed frames with detection overlays

### Webcam Mode
1. Navigate to **AI Monitoring** page
2. Select **Live Webcam** mode
3. Select camera indices for front and side views
4. Click **Start Monitoring**
5. View real-time contact detection

### Features
- Real-time frame streaming via WebSocket
- Contact collision alerts displayed as notifications
- Configurable detection parameters (confidence, threshold, etc.)
- Session status tracking
- GPU acceleration support

---

## Contact Monitoring (Legacy CLI)

The monitoring system can still run via command line:

```powershell
python src/monitor_contacts.py
```

This will:
- Use dual cameras for person tracking
- Detect close contacts between individuals
- Store contact data in MongoDB
- Send email alerts for MDR contacts
- Display real-time alerts in the web interface

---

## Configuration (.env)

Key settings in `.env`:

```env
# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=patient_contact_tracing

# JWT Authentication
JWT_SECRET_KEY=your-secret-key-min-32-chars
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Email Alerts
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Face Registration
FACE_REGISTRATION_TOTAL_SAMPLES=50
FACE_RECOGNITION_THRESHOLD=0.35
```

---

## Legacy CLI Commands

### Register a Person (CLI)

```powershell
python src/register_face.py register "Person Name"
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

### 3. Mark/Unmark MDR Patients

Designate patients as MDR (Multi-Drug Resistant) to trigger email alerts when they contact others:

```powershell
# Mark a patient as MDR
C:/Users/mourish/Desktop/sih_01/venv/Scripts/python.exe src/mark_mdr.py mark "Patient Name"

# Remove MDR status
C:/Users/mourish/Desktop/sih_01/venv/Scripts/python.exe src/mark_mdr.py unmark "Patient Name"

# List all MDR patients
C:/Users/mourish/Desktop/sih_01/venv/Scripts/python.exe src/mark_mdr.py list

# Check if a patient is MDR
C:/Users/mourish/Desktop/sih_01/venv/Scripts/python.exe src/mark_mdr.py check "Patient Name"
```

**MDR Email Alerts:**
- When an MDR patient contacts another person, the system monitors the duration
- If contact reaches 5 minutes (configurable), an email alert is sent immediately with snapshots
- If contact ends before 5 minutes, no alert is sent
- If contact exceeds 5 minutes and then ends, a completion alert is sent (if not already sent)
- Alerts include: MDR patient name, contacted person, timestamps, duration, risk %, camera snapshots

**Email Configuration:**
Configure SMTP settings in `.env`:
```env
MDR_ALERT_THRESHOLD_SECONDS=300  # 5 minutes
MDR_SMTP_SERVER=smtp.gmail.com
MDR_SMTP_PORT=587
MDR_SMTP_USERNAME=your-email@gmail.com
MDR_SMTP_PASSWORD=your-app-password
MDR_ADMIN_EMAIL=admin@hospital.com
```

For Gmail, you need to create an [App Password](https://support.google.com/accounts/answer/185833).

### 4. Monitor Contacts (Dual-Camera Contact Tracing)

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

**Risk Levels:**
| Level | Risk Percentage | Color |
|-------|----------------|-------|
| Low | 0% - 39% | Green |
| Medium | 40% - 69% | Orange |
| High | 70% - 100% | Red |

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
- **MDR patient list:** `data/mdr_patients.json` (delete to clear MDR flags)
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
2. **Mark MDR patients** using `mark_mdr.py mark "Name"` (optional but recommended for MDR tracking)
3. **Monitor contacts** using `monitor_contacts.py`:
   - InsightFace recognizes faces (even with masks)
   - YOLO + DeepSORT tracks full body movement
   - Collision detector measures proximity using IoU + distance
   - Risk accumulates over time with mask modifiers
   - Alerts trigger when thresholds exceeded
   - Contact logs record timestamps + risk percentages
   - **MDR email alerts** sent when MDR patient contacts others for ≥5 minutes
4. **Review data** in `Contact_Details/` and `data/alerts/`
5. **Check inbox** for MDR alert emails (if configured)
6. **Unregister patients** when needed using `register_face.py unregister "Name"`
7. **Remove MDR status** when needed using `mark_mdr.py unmark "Name"`

## Technology Stack

- **Face Recognition:** InsightFace (ArcFace), ONNX Runtime
- **Person Detection:** YOLO v8
- **Person Tracking:** DeepSORT + OSNet
- **Mask Classification:** Custom CNN (trained on Kaggle dataset)
- **Collision Detection:** IoU + Euclidean distance with risk scoring
- **Alert System:** Duration-gated with cooldown periods
- **UI:** OpenCV + Rich console output
