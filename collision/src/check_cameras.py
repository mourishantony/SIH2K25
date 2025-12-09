import cv2

print("Checking available cameras...")
print()

available_cameras = []

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"✓ Camera {i}: {width}x{height} @ {fps:.1f}fps")
            available_cameras.append(i)
        cap.release()

print()
if len(available_cameras) == 0:
    print("No cameras found!")
elif len(available_cameras) == 1:
    print(f"Found 1 camera (index {available_cameras[0]})")
    print("\nTo use with dual-camera system:")
    print("- You need 2 cameras connected")
    print("- OR use one camera with a video file:")
    print(f'  python src/monitor_collision.py --camera1 {available_cameras[0]} --video2 "path/to/video.mp4"')
else:
    print(f"Found {len(available_cameras)} cameras: {available_cameras}")
    print("\nTo use both cameras:")
    print(f"  python src/monitor_collision.py --camera1 {available_cameras[0]} --camera2 {available_cameras[1]}")
