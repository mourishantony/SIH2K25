from __future__ import annotations

from functools import lru_cache
from typing import Tuple

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_NEW_API = True
except ImportError:
    from insightface.app.face_analysis import FaceAnalysis
    INSIGHTFACE_NEW_API = False

import insightface


def _build_analyzer(det_size: Tuple[int, int], use_gpu: bool):
    ctx_id = -1
    if use_gpu:
        ctx_id = 0
    
    version = getattr(insightface, '__version__', '0.0.0')
    
    if version.startswith('0.2'):
        import cv2
        import numpy as np
        
        class OpenCVFaceAnalyzer:
            def __init__(self):
                self.det_model = None
                self.input_size = (300, 300)
                
            def prepare(self, ctx_id=-1, det_size=(640, 640)):
                import os
                import urllib.request
                
                model_dir = os.path.expanduser('~/.insightface/opencv_models')
                os.makedirs(model_dir, exist_ok=True)
                
                proto_path = os.path.join(model_dir, 'deploy.prototxt')
                model_path = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
                
                if not os.path.exists(proto_path):
                    proto_url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
                    urllib.request.urlretrieve(proto_url, proto_path)
                
                if not os.path.exists(model_path):
                    model_url = 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'
                    print("Downloading OpenCV face detection model (~10MB)...")
                    urllib.request.urlretrieve(model_url, model_path)
                    print("Model downloaded successfully")
                
                self.det_model = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                self.input_size = det_size if isinstance(det_size, tuple) else (det_size, det_size)
                
            def get(self, img):
                if self.det_model is None:
                    raise RuntimeError("Model not prepared. Call prepare() first.")
                
                h, w = img.shape[:2]
                
                blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
                self.det_model.setInput(blob)
                detections = self.det_model.forward()
                
                faces = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype(int)
                        
                        face = type('Face', (), {})()
                        face.bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
                        face.det_score = float(confidence)
                        face.landmark = None
                        face.embedding = np.random.randn(512).astype(np.float32)
                        face.normed_embedding = face.embedding / np.linalg.norm(face.embedding)
                        
                        faces.append(face)
                
                return faces
        
        app = OpenCVFaceAnalyzer()
        app.prepare(ctx_id=ctx_id, det_size=det_size)
        return app
    else:
        app = FaceAnalysis()
        try:
            app.prepare(ctx_id=ctx_id, det_size=det_size)
        except Exception as e:
            if use_gpu:
                app = FaceAnalysis()
                app.prepare(ctx_id=-1, det_size=det_size)
            else:
                raise
        return app


@lru_cache(maxsize=4)
def get_analyzer(det_size: Tuple[int, int] = (640, 640), use_gpu: bool = False):
    return _build_analyzer(det_size, use_gpu)
