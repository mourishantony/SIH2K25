import cv2
import numpy as np
from vision import get_analyzer

def test_video_detection(video_path, max_frames=100):
    print(f"Testing: {video_path}")
    
    analyzer = get_analyzer((640, 640), use_gpu=False)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return
    
    frame_count = 0
    faces_detected = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        faces = analyzer.get(frame)
        
        if len(faces) > 0:
            faces_detected += 1
            print(f"Frame {frame_count}: Found {len(faces)} face(s)")
            
            for face in faces:
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {face.det_score:.2f}", 
                           (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
        
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        
        cv2.imshow(f"Detection Test - {video_path}", frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nResults for {video_path}:")
    print(f"  Total frames: {frame_count}")
    print(f"  Frames with faces: {faces_detected}")
    print(f"  Detection rate: {faces_detected/frame_count*100:.1f}%\n")

if __name__ == "__main__":
    print("Face Detection Test")
    print("=" * 50)
    print("This will show you what the face detector sees.")
    print("Press 'q' to skip to next video.\n")
    
    # Test both videos
    test_video_detection("C:/Users/Tayanithaa.N.S/neura5/sih_01/src/vid1.mp4", max_frames=100)
    test_video_detection("C:/Users/Tayanithaa.N.S/neura5/sih_01/src/vid2.mp4", max_frames=100)
    
    print("\nTest complete!")
    print("\nIf no faces were detected:")
    print("  1. Videos might not contain people")
    print("  2. People might be too far from camera")
    print("  3. Videos might be drone/aerial footage")
    print("  4. Face detector optimized for frontal faces")
