import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from loguru import logger
from core.face_quality import FaceQualityAssessor
from core.face_alignment import FaceAligner
from core.schemas import FaceQualityResult, RecognitionReadyFace

class FaceProcessor:
    """
    Phase 5 Unified Processing Pipeline:
    Connects detector outputs to recognition readiness.
    Performs Quality Assessment, 5-point Alignment, Preprocessing/Normalization,
    and returns standardized RecognitionReadyFace objects.
    """
    def __init__(self, assessor: Optional[FaceQualityAssessor] = None, aligner: Optional[FaceAligner] = None):
        self.assessor = assessor or FaceQualityAssessor()
        self.aligner = aligner or FaceAligner()

    def process(
        self,
        detections: List[Dict[str, Any]],
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> Tuple[List[RecognitionReadyFace], List[FaceQualityResult]]:
        """
        Process multiple detected faces from a frame.
        
        Args:
            detections: List of detection dictionaries (with bbox, face_area, keypoints)
            frame: Input BGR image frame
            timestamp: Optional epoch timestamp (defaults to current time)
            
        Returns:
            Tuple of:
              - ready_faces: List[RecognitionReadyFace] for Phase 6 ArcFace
              - rejected_faces: List[FaceQualityResult] for logging/monitoring
        """
        ts = timestamp if timestamp is not None else time.time()
        ready_faces: List[RecognitionReadyFace] = []
        rejected_faces: List[FaceQualityResult] = []

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return ready_faces, rejected_faces

        for i, det in enumerate(detections):
            face_id = det.get("track_id") or det.get("face_id") or f"face_{i}_{int(ts * 1000)}"
            quality_result = self.assessor.assess(det, frame, str(face_id))

            if quality_result.quality_status == "RECOGNITION_READY" and quality_result.keypoints is not None and len(quality_result.keypoints) == 5:
                try:
                    tensor = self.aligner.align_and_normalize(frame, quality_result.keypoints)
                    ready_face = RecognitionReadyFace(
                        quality_metrics=quality_result,
                        original_bbox=det.get("bbox", quality_result.bbox),
                        aligned_face_tensor=tensor,
                        timestamp=ts
                    )
                    ready_faces.append(ready_face)
                except Exception as e:
                    logger.warning(f"Face alignment/normalization failed for {face_id}: {e}")
                    quality_result.quality_status = "LOW_QUALITY"
                    quality_result.rejection_reason = f"Alignment error: {str(e)}"
                    rejected_faces.append(quality_result)
            else:
                rejected_faces.append(quality_result)

        return ready_faces, rejected_faces
