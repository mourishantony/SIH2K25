class GPUCollisionSettings:
    
    use_gpu = True
    gpu_device = "cuda"
    enable_fp16 = True
    gpu_memory_fraction = 0.8
    
    yolo_model_size = "s"
    yolo_image_size = 640
    yolo_confidence = 0.25
    yolo_max_detections = 50
    
    enable_tensorrt = False
    batch_size = 1
    enable_benchmark = True
    
    iou_threshold = 0.1
    distance_threshold = 200
    min_confidence = 0.3
    det_size = 640
    
    max_fps = 60
    frame_skip = 0
    gpu_warmup_frames = 5
    
    clear_cache_frequency = 100
    enable_memory_monitoring = True
    
    enable_audio = True
    enable_logging = True
    min_risk_for_alert = "MEDIUM"
    min_alert_interval = 2.0
    
    camera1_index = 0
    camera2_index = 1
    recognition_threshold = 0.5
    require_both_cameras = True
    
    display_mode = "side"
    max_display_width = 960
    show_fps = True
    show_gpu_stats = True


gpu_settings = GPUCollisionSettings()