import cv2
import sys
import os
import time
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.camera_manager import CameraManager
from core.detector import YOLOv8FaceDetector
from core.face_quality import FaceQualityAssessor
from core.face_alignment import FaceAligner
from loguru import logger

def run_live_detection():
    print("Starting Live Detection & Quality Preview...")
    print("Press 's' to save aligned crops to debug/")
    print("Press 'q' to quit.")
    
    debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    manager = CameraManager()
    
    try:
        detector = YOLOv8FaceDetector(model_path="yolov8n-face.pt") 
        assessor = FaceQualityAssessor()
        aligner = FaceAligner()
    except Exception as e:
        logger.error(f"Could not load modules: {e}")
        return

    window_name = "SmartClass Vision AI - Quality Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    fps_start_time = time.time()
    fps_frame_count = 0
    display_fps = 0.0
    
    save_next_frame = False
    
    while True:
        success, frame = manager.get_frame(max_retries=3)
        
        if not success:
            print("Failed to get frame. Exiting preview.")
            break
            
        # Run detection
        detections = detector.detect(frame)
        faces = detections

        
        saved_count = 0
        
        for i, face in enumerate(faces):
            # Phase 5: Quality Assessment
            face_id = f"face_{i}"
            quality_result = assessor.assess(face, frame, face_id)
            
            x1, y1, x2, y2 = quality_result.bbox
            
            color = (0, 255, 0) if quality_result.quality_status == "RECOGNITION_READY" else (0, 0, 255)
            
            # Phase 5: Alignment & Normalization
            if quality_result.quality_status == "RECOGNITION_READY" and quality_result.keypoints:
                try:
                    aligned_face = aligner.align(frame, quality_result.keypoints)
                    # We don't save the normalized tensor to disk because it's a float32 array
                    # but we can save the aligned RGB/BGR crop for visual debugging
                    if save_next_frame:
                        out_path = os.path.join(debug_dir, f"{uuid.uuid4().hex[:8]}.jpg")
                        cv2.imwrite(out_path, aligned_face)
                        saved_count += 1
                except Exception as e:
                    logger.error(f"Alignment failed: {e}")
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw labels (Size, Sharpness, Quality)
            label1 = f"Size: {quality_result.width}x{quality_result.height}"
            label2 = f"Sharp: {int(quality_result.sharpness)} | Br: {int(quality_result.brightness)}"
            label3 = quality_result.quality_status
            
            cv2.putText(frame, label1, (x1, max(20, y1 - 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(frame, label2, (x1, max(20, y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(frame, label3, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw keypoints if available
            if quality_result.keypoints:
                for pt in quality_result.keypoints:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)
            
        if save_next_frame:
            logger.info(f"Saved {saved_count} aligned crops to debug folder.")
            save_next_frame = False
            
        # Calculate FPS
        fps_frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed > 1.0:
            display_fps = fps_frame_count / elapsed
            fps_start_time = time.time()
            fps_frame_count = 0
            
        status = manager.get_status()
        
        # Overlay diagnostic text
        font = cv2.FONT_HERSHEY_SIMPLEX
        y0, dy = 30, 30
        texts = [
            f"Faces Detected: {len(faces)}",
            f"Processing FPS: {display_fps:.1f}",
            f"Camera FPS: {status.get('measured_fps', 0.0)}"
        ]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 10 + len(texts)*dy + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        for i, text in enumerate(texts):
            y = y0 + i * dy
            cv2.putText(frame, text, (20, y), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_next_frame = True
            
    manager.release()
    cv2.destroyAllWindows()
    print("Live detection closed.")

if __name__ == "__main__":
    run_live_detection()
