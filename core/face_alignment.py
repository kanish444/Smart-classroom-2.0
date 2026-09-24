import cv2
import numpy as np
from typing import List, Tuple, Any

class FaceAligner:
    """
    Performs 5-point affine transformation and normalization 
    required for ArcFace (InsightFace) recognition models.
    """
    # Standard reference keypoints for 112x112 image required by ArcFace
    REFERENCE_FACIAL_POINTS = np.array([
        [38.2946, 51.6963],  # Left Eye
        [73.5318, 51.5014],  # Right Eye
        [56.0252, 71.7366],  # Nose
        [41.5493, 92.3655],  # Left Mouth
        [70.7299, 92.2041]   # Right Mouth
    ], dtype=np.float32)

    def __init__(self, output_size: Tuple[int, int] = (112, 112)):
        self.output_size = output_size
        
    def align(self, frame: np.ndarray, keypoints: List[List[float]]) -> np.ndarray:
        """
        Calculates and applies the affine transformation matrix to align the face.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid or empty frame passed to align.")
            
        if not keypoints or len(keypoints) != 5:
            raise ValueError("Exactly 5 keypoints are required for alignment.")
            
        src_pts = np.array(keypoints, dtype=np.float32)
        dst_pts = self.REFERENCE_FACIAL_POINTS
        
        # Calculate similarity transform (uses RANSAC internally if enough points, but for 5 points it uses all)
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        
        if M is None:
            # Fallback if math fails
            raise ValueError("Failed to estimate affine transformation matrix.")
            
        # Warp the image
        aligned_face = cv2.warpAffine(
            frame, 
            M, 
            self.output_size, 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0, 0, 0)
        )
        
        return aligned_face
        
    def normalize(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Normalizes the 112x112 BGR face into the tensor format expected by MobileFaceNet/ArcFace.
        Standard ArcFace preprocessing: (img - 127.5) / 128.0
        """
        if aligned_face is None or not isinstance(aligned_face, np.ndarray) or aligned_face.size == 0:
            raise ValueError("Invalid or empty aligned face passed to normalize.")
            
        # Convert BGR to RGB (ArcFace expects RGB)
        rgb_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 and normalize
        tensor_face = (rgb_face.astype(np.float32) - 127.5) / 128.0
        
        # Channel-first format: (H, W, C) -> (C, H, W)
        tensor_face = np.transpose(tensor_face, (2, 0, 1))
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        tensor_face = np.expand_dims(tensor_face, axis=0)
        
        return tensor_face

    def align_and_normalize(self, frame: np.ndarray, keypoints: List[List[float]]) -> np.ndarray:
        """
        Convenience method executing both 5-point alignment and tensor normalization.
        """
        aligned = self.align(frame, keypoints)
        return self.normalize(aligned)
