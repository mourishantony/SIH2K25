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
   - `FACE_RECOG_*` entries control the streaming script (detection confidence, similarity threshold, detector size, camera index, GPU flag).
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
C:/Users/mourish/Desktop/sih_01/venv/Scripts/python.exe src/register_face.py "Person Name" --use-gpu
```

Drop `--use-gpu` if you prefer to stay on the CPU runtime.

- Phase 1 runs with the mask **off**. Keep your face centered; samples are taken automatically every ~0.45s.
- After Phase 1, the script pauses. Put on your mask and press `Enter` in the terminal to resume Phase 2.
- Press `q` in the preview window at any time to abort.

You can tweak options such as `--unmasked-samples`, `--masked-samples`, `--camera-index`, and `--capture-delay` if needed.

## Real-Time Recognition

```powershell
C:/Users/mourish/Desktop/sih_01/venv/Scripts/python.exe src/recognize_face.py watch --use-gpu
```

Remove `--use-gpu` to keep the workload on the CPU.

To change the default similarity guard, set `FACE_RECOG_THRESHOLD` inside `.env` (CLI `--threshold` still wins when supplied).

- The stream window overlays bounding boxes and the predicted identity with similarity score.
- Press `q` to exit. Use `--threshold` to make recognition stricter or more permissive.

## Data Storage

Embeddings are written to `data/face_database.json`. You can safely delete this file to reset the registry.

## Troubleshooting

- The first run downloads InsightFace models; keep an active internet connection.
- If detection is unreliable, increase lighting and adjust `--min-confidence`.
- For best mask performance, capture balanced samples with and without the mask and consider using different mask colors/forms during registration.
- If you see InsightFace fall back to CPU, double-check that `onnxruntime-gpu` is installed and that the NVIDIA driver exposes CUDA to ONNX Runtime.
