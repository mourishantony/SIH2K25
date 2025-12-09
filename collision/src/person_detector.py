import cv2
import numpy as np
from typing import List, Tuple
import os
import urllib.request


class PersonDetector:
    
    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.net = None
        self.output_layers = None
        self.classes = None
        self._load_model()
    
    def _load_model(self):
        model_dir = os.path.expanduser('~/.insightface/yolo_models')
        os.makedirs(model_dir, exist_ok=True)
        
        weights_path = os.path.join(model_dir, 'yolov3.weights')
        config_path = os.path.join(model_dir, 'yolov3.cfg')
        names_path = os.path.join(model_dir, 'coco.names')
        
        if not os.path.exists(weights_path):
            print("Downloading YOLOv3 weights (~240MB)...")
            urllib.request.urlretrieve(
                'https://pjreddie.com/media/files/yolov3.weights',
                weights_path
            )
            print("Weights downloaded")
        
        if not os.path.exists(config_path):
            print("Downloading YOLOv3 config...")
            urllib.request.urlretrieve(
                'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
                config_path
            )
        
        if not os.path.exists(names_path):
            print("Downloading COCO class names...")
            urllib.request.urlretrieve(
                'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names',
                names_path
            )
        
        self.net = cv2.dnn.readNet(weights_path, config_path)
        
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("✓ Using GPU acceleration for YOLO")
        except:
            print("Using CPU for YOLO (GPU not available)")
        
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
    
    def detect(self, frame: np.ndarray) -> List[Tuple[np.ndarray, float, str]]:
        height, width = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if class_id == 0 and confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x1 = int(center_x - w / 2)
                    y1 = int(center_y - h / 2)
                    x2 = int(center_x + w / 2)
                    y2 = int(center_y + h / 2)
                    
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)
                    
                    boxes.append([x1, y1, w, h])
                    confidences.append(float(confidence))
        
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
            
            results = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x1, y1, w, h = boxes[i]
                    x2 = x1 + w
                    y2 = y1 + h
                    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
                    results.append((bbox, confidences[i], f"Person_{i}"))
            
            return results
        
        return []


def get_person_detector(confidence_threshold: float = 0.3) -> PersonDetector:
    return PersonDetector(confidence_threshold)
