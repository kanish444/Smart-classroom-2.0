import sys
import os
import cv2
import numpy as np
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.face_quality import FaceQualityAssessor
from core.face_alignment import FaceAligner
from core.schemas import RecognitionReadyFace

def create_synthetic_face(w: int, h: int, color: tuple, blur: bool = False) -> np.ndarray:
    """Create a mock face with realistic facial features for testing."""
    face = np.full((h, w, 3), color, dtype=np.uint8)
    b_val = float(np.mean(color))
    feat_color = (0, 0, 0) if b_val > 40 else (5, 5, 5)
    
    # Facial features (eyes, nose, mouth)
    eye_radius = max(2, int(min(w, h) * 0.06))
    left_eye = (int(w * 0.35), int(h * 0.35))
    right_eye = (int(w * 0.65), int(h * 0.35))
    cv2.circle(face, left_eye, eye_radius, feat_color, -1)
    cv2.circle(face, right_eye, eye_radius, feat_color, -1)
    cv2.line(face, (int(w * 0.5), int(h * 0.45)), (int(w * 0.5), int(h * 0.6)), feat_color, max(1, int(w * 0.03)))
    cv2.rectangle(face, (int(w * 0.35), int(h * 0.75)), (int(w * 0.65), int(h * 0.82)), feat_color, -1)
    
    if blur:
        ksize = max(15, (w // 8) * 2 + 1)
        face = cv2.GaussianBlur(face, (ksize, ksize), 0)
    return face

def run_tests():
    logger.info("==================================================")
    logger.info("PHASE 5: QUALITY GATING & ALIGNMENT TESTS")
    logger.info("==================================================")
    
    assessor = FaceQualityAssessor()
    aligner = FaceAligner()
    
    frame_w, frame_h = 1920, 1080
    mock_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    
    # Define test cases: (name, bbox, color, is_blurry, expected_status)
    test_cases = [
        ("TEST 1 - Normal High Quality Face", [100, 100, 300, 350], (200, 200, 200), False, "RECOGNITION_READY"),
        ("TEST 2 - Very Small/Far Face", [10, 10, 40, 50], (200, 200, 200), False, "LOW_QUALITY"),
        ("TEST 3 - Blurry Face", [400, 400, 600, 650], (200, 200, 200), True, "LOW_QUALITY"),
        ("TEST 4 - Dark Face", [100, 500, 300, 750], (20, 20, 20), False, "LOW_QUALITY"),
        ("TEST 5 - Bright Face (Glare)", [500, 100, 700, 350], (250, 250, 250), False, "LOW_QUALITY"),
        ("TEST 6 - Face near boundary", [1850, 1000, 1950, 1100], (200, 200, 200), False, "RECOGNITION_READY"),
    ]
    
    results = {"passed": 0, "failed": 0}
    
    for name, bbox, color, is_blurry, expected_status in test_cases:
        x1, y1, x2, y2 = bbox
        
        # Draw on mock frame
        safe_x1, safe_y1 = max(0, x1), max(0, y1)
        safe_x2, safe_y2 = min(frame_w, x2), min(frame_h, y2)
        mock_face = create_synthetic_face(safe_x2 - safe_x1, safe_y2 - safe_y1, color, is_blurry)
        mock_frame[safe_y1:safe_y2, safe_x1:safe_x2] = mock_face
        
        area = (x2 - x1) * (y2 - y1)
        
        # Mock detector output
        mock_detection = {
            "bbox": bbox,
            "face_area": area,
            "keypoints": [
                [x1 + 20, y1 + 30], [x1 + 60, y1 + 30], 
                [x1 + 40, y1 + 50], 
                [x1 + 30, y1 + 80], [x1 + 50, y1 + 80]
            ]
        }
        
        # 1. Quality Assessment
        quality_res = assessor.assess(mock_detection, mock_frame, face_id="test_face")
        
        logger.info(f"--- {name} ---")
        logger.info(f"Expected: {expected_status} | Actual: {quality_res.quality_status}")
        logger.info(f"Reason: {quality_res.rejection_reason}")
        logger.info(f"Metrics -> Area: {quality_res.area}, Sharpness: {quality_res.sharpness:.1f}, Brightness: {quality_res.brightness:.1f}")
        
        if quality_res.quality_status == expected_status:
            logger.success("PASS")
            results["passed"] += 1
        else:
            logger.error("FAIL")
            results["failed"] += 1
            
        # 2. Alignment & RecognitionReadyFace Packaging Test (only on ready faces)
        if quality_res.quality_status == "RECOGNITION_READY":
            try:
                aligned_crop = aligner.align(mock_frame, quality_res.keypoints)
                normalized_tensor = aligner.normalize(aligned_crop)
                logger.info(f"Alignment Success: Tensor Shape = {normalized_tensor.shape}")
                
                # Verify tensor shape (1, 3, 112, 112)
                if normalized_tensor.shape == (1, 3, 112, 112):
                    ready_face = RecognitionReadyFace(
                        quality_metrics=quality_res,
                        original_bbox=bbox,
                        aligned_face_tensor=normalized_tensor,
                        timestamp=1000.0
                    )
                    assert ready_face.aligned_face_tensor.shape == (1, 3, 112, 112)
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                logger.error(f"Alignment Failed: {e}")
                results["failed"] += 1
                
    logger.info("==================================================")
    logger.info(f"TEST RUN COMPLETED: {results['passed']} Passed, {results['failed']} Failed")
    logger.info("==================================================")

if __name__ == "__main__":
    run_tests()
