import cv2
import numpy as np
from typing import Dict, Any, Tuple
from core.schemas import FaceQualityResult
from config.settings import get_settings

class FaceQualityAssessor:
    """
    Evaluates raw bounding box crops for sufficient quality (sharpness, brightness, size)
    before they are passed to the alignment and recognition pipeline.
    """
    def __init__(self):
        self.settings = get_settings().quality
        
    def evaluate_sharpness(self, crop: np.ndarray) -> float:
        """
        Calculates the Variance of the Laplacian.
        Higher is sharper.
        """
        if crop is None or crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
    def evaluate_brightness(self, crop: np.ndarray) -> float:
        """
        Calculates mean pixel intensity of the crop.
        0 is black, 255 is white.
        """
        if crop is None or crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 2]))
        
    def assess(self, face_dict: Dict[str, Any], frame: np.ndarray, face_id: str) -> FaceQualityResult:
        """
        Assess a single detection dictionary and return the final decision.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return FaceQualityResult(
                face_id=face_id,
                bbox=[0, 0, 0, 0],
                keypoints=None,
                width=0,
                height=0,
                area=0,
                sharpness=0.0,
                brightness=0.0,
                pose="UNKNOWN",
                quality_status="LOW_QUALITY",
                rejection_reason="Invalid or empty image frame"
            )

        bbox = face_dict.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        area = face_dict.get('face_area', (x2 - x1) * (y2 - y1))
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        
        # Ensure bounds
        x1_safe = max(0, x1)
        y1_safe = max(0, y1)
        x2_safe = min(frame.shape[1], x2)
        y2_safe = min(frame.shape[0], y2)
        
        if x1_safe >= x2_safe or y1_safe >= y2_safe:
            return FaceQualityResult(
                face_id=face_id,
                bbox=[x1_safe, y1_safe, x2_safe, y2_safe],
                keypoints=face_dict.get('keypoints'),
                width=w,
                height=h,
                area=area,
                sharpness=0.0,
                brightness=0.0,
                pose="UNKNOWN",
                quality_status="LOW_QUALITY",
                rejection_reason="Bounding box outside frame boundaries or degenerate crop"
            )
            
        crop = frame[y1_safe:y2_safe, x1_safe:x2_safe]
        
        # Calculate Metrics
        sharpness = self.evaluate_sharpness(crop)
        brightness = self.evaluate_brightness(crop)
        
        # Gate Logic
        status = "RECOGNITION_READY"
        reason = None
        
        if area < self.settings.min_face_area:
            status = "LOW_QUALITY"
            reason = f"Too small ({area} < {self.settings.min_face_area} px)"
        elif brightness < self.settings.min_brightness:
            status = "LOW_QUALITY"
            reason = f"Too dark ({brightness:.1f} < {self.settings.min_brightness})"
        elif brightness > self.settings.max_brightness:
            status = "LOW_QUALITY"
            reason = f"Too bright ({brightness:.1f} > {self.settings.max_brightness})"
        elif sharpness < self.settings.min_sharpness:
            status = "LOW_QUALITY"
            reason = f"Too blurry ({sharpness:.1f} < {self.settings.min_sharpness})"
            
        return FaceQualityResult(
            face_id=face_id,
            bbox=[x1_safe, y1_safe, x2_safe, y2_safe],
            keypoints=face_dict.get('keypoints'),
            width=w,
            height=h,
            area=area,
            sharpness=sharpness,
            brightness=brightness,
            pose="UNKNOWN",
            quality_status=status,
            rejection_reason=reason
        )
