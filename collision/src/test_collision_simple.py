import cv2
from multi_camera import DualCameraManager
from collision_detector import detect_collisions

camera_manager = DualCameraManager(
    source1="src/vid4.mp4",
    source2="src/vid4_1.mp4",
    det_size=(640, 640),
    use_gpu=False,
    min_confidence=0.3,
    threshold=0.5,
)

dims1, dims2 = camera_manager.get_frame_dimensions()
print(f"Camera 1: {dims1[0]}x{dims1[1]}")
print(f"Camera 2: {dims2[0]}x{dims2[1]}")
print()

frame_count = 0
total_collisions_cam1 = 0
total_collisions_cam2 = 0

while frame_count < 100:
    success, frame1, frame2 = camera_manager.read_synchronized_frames()
    if not success:
        break
    
    frame_count += 1
    
    bboxes1, bboxes2 = camera_manager.process_frames(frame1, frame2)
    
    collisions1 = detect_collisions(
        bboxes1,
        iou_threshold=0.01,
        distance_threshold=800,
        frame_width=dims1[0],
        frame_height=dims1[1],
    )
    
    collisions2 = detect_collisions(
        bboxes2,
        iou_threshold=0.01,
        distance_threshold=800,
        frame_width=dims2[0],
        frame_height=dims2[1],
    )
    
    if collisions1 or collisions2:
        print(f"Frame {frame_count}:")
        if collisions1:
            total_collisions_cam1 += len(collisions1)
            print(f"  Camera 1: {len(collisions1)} collisions")
            for c in collisions1:
                print(f"    - {c.person1} <-> {c.person2}: Risk={c.risk_level} ({c.risk_score:.2%}), IoU={c.iou:.3f}")
        if collisions2:
            total_collisions_cam2 += len(collisions2)
            print(f"  Camera 2: {len(collisions2)} collisions")
            for c in collisions2:
                print(f"    - {c.person1} <-> {c.person2}: Risk={c.risk_level} ({c.risk_score:.2%}), IoU={c.iou:.3f}")
        print()

print(f"\nSummary:")
print(f"Frames processed: {frame_count}")
print(f"Total collisions in Camera 1: {total_collisions_cam1}")
print(f"Total collisions in Camera 2: {total_collisions_cam2}")
