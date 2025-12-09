
import cv2
from hybrid_detector import get_hybrid_detector

def debug_detection(video_path, max_frames=50):
    detector = get_hybrid_detector(min_confidence=0.3)
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    max_people_in_frame = 0
    frames_with_multiple_people = 0
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {video_path}")
    print(f"{'='*60}\n")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        detections = detector.detect(frame)
        
        if len(detections) > 0:
            print(f"Frame {frame_count}: {len(detections)} people detected")
            for bbox, conf, identity in detections:
                x1, y1, x2, y2 = bbox.astype(int)
                w = x2 - x1
                h = y2 - y1
                print(f"  - {identity}: bbox=({x1},{y1},{x2},{y2}) size={w}x{h} conf={conf:.2f}")
            
            if len(detections) > 1:
                frames_with_multiple_people += 1
             
                for i in range(len(detections)):
                    for j in range(i+1, len(detections)):
                        bbox1 = detections[i][0]
                        bbox2 = detections[j][0]
                        
                       
                        c1_x = (bbox1[0] + bbox1[2]) / 2
                        c1_y = (bbox1[1] + bbox1[3]) / 2
                        c2_x = (bbox2[0] + bbox2[2]) / 2
                        c2_y = (bbox2[1] + bbox2[3]) / 2
                        
                        import numpy as np
                        distance = np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
                        
                        print(f"  >> Distance between person {i} and {j}: {distance:.0f}px")
            
            max_people_in_frame = max(max_people_in_frame, len(detections))
    
    cap.release()
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total frames analyzed: {frame_count}")
    print(f"  Max people in single frame: {max_people_in_frame}")
    print(f"  Frames with 2+ people: {frames_with_multiple_people}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    print("\n🔍 DETECTION DEBUG TOOL 🔍\n")
    debug_detection("src/vid1.mp4", max_frames=100)
    debug_detection("src/vid2.mp4", max_frames=100)
