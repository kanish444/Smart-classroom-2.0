import sys
import os
import time
import cv2
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.detector import YOLOv8FaceDetector
from camera.camera_manager import CameraManager

def evaluate_detector(image_path: str = None):
    logger.info("=" * 50)
    logger.info("PHASE 4: STRESS TEST BENCHMARK")
    logger.info("=" * 50)
    
    frame = None
    if image_path and os.path.exists(image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error("Failed to load test image.")
            return
        h, w = frame.shape[:2]
        logger.info(f"Loaded test image: {w}x{h}")
    else:
        logger.info("No valid test image provided. Attempting to capture from Live Camera...")
        manager = CameraManager()
        success, frame = manager.get_frame(max_retries=3)
        manager.release()
        if not success or frame is None:
            logger.error("Failed to capture frame from live camera.")
            return
        h, w = frame.shape[:2]
        logger.info(f"Captured live frame: {w}x{h}")
    
    try:
        detector = YOLOv8FaceDetector(model_path="yolov8n-face.pt") 
    except Exception as e:
        logger.error(f"Detector failed to load: {e}")
        return
        
    # Warmup
    logger.info("Warming up model...")
    detector.detect(frame)
    
    # Run Benchmark
    logger.info("Running detection...")
    t0 = time.time()
    detections = detector.detect(frame)
    t1 = time.time()
    
    # Authentic tight face detections
    faces = detections

    
    latency_ms = (t1 - t0) * 1000
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0
    
    zones = {"NEAR": 0, "MIDDLE": 0, "FAR": 0}
    sizes = []
    
    for face in faces:
        zones[face['zone']] += 1
        sizes.append(face['face_area'])
        
        # Draw on frame for visual validation output
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    min_size = min(sizes) if sizes else 0
    max_size = max(sizes) if sizes else 0
    
    logger.info("=" * 30)
    logger.info(f"Total Faces Detected : {len(faces)}")
    logger.info(f"Inference Latency  : {latency_ms:.2f} ms")
    logger.info(f"Estimated FPS      : {fps:.2f}")
    if sizes:
        logger.info(f"Average Face Area  : {avg_size:.1f} px")
        logger.info(f"Smallest Face Area : {min_size} px")
        logger.info(f"Largest Face Area  : {max_size} px")
    logger.info("--- Zones ---")
    logger.info(f"NEAR   (>15k px)   : {zones['NEAR']}")
    logger.info(f"MIDDLE (3k-15k px) : {zones['MIDDLE']}")
    logger.info(f"FAR    (<3k px)    : {zones['FAR']}")
    logger.info("=" * 30)
    
    out_path = "stress_test_result.jpg"
    cv2.imwrite(out_path, frame)
    logger.info(f"Saved visual result to {out_path}")

if __name__ == "__main__":
    test_img = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate_detector(test_img)
