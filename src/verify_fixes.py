import sys, os
sys.path.insert(0, 'src')
sys.path.insert(0, 'backend')
errors = []

# Check 1: reid_tracker patch
try:
    import reid_tracker
    if not hasattr(reid_tracker, '_patch_torch_load_for_ultralytics'):
        errors.append('reid_tracker.py patch NOT found')
    else:
        print('[OK] reid_tracker.py patch is active')
except Exception as e:
    errors.append(f'reid_tracker import failed: {e}')

# Check 2: YOLO loads
try:
    from ultralytics import YOLO
    from pathlib import Path
    pt = Path('src/yolov8n.pt')
    if pt.exists():
        try:
            YOLO(str(pt))
            print('[OK] YOLO model loads successfully')
        except Exception as e:
            errors.append(f'YOLO load failed: {e}')
    else:
        errors.append('src/yolov8n.pt not found')
except Exception as e:
    errors.append(f'ultralytics import failed: {e}')

# Check 3: mask_detector.joblib deleted
if not os.path.exists('src/mask_datas/mask_detector.joblib'):
    print('[OK] Stale mask_detector.joblib removed')
else:
    print('[WARN] mask_detector.joblib still exists (will warn but not crash)')

# Check 4: sklearn version
import sklearn
print(f'[OK] scikit-learn version: {sklearn.__version__}')

if errors:
    print()
    for e in errors:
        print(f'[ERROR] {e}')
else:
    print()
    print('All checks passed! Restart your FastAPI backend now.')
