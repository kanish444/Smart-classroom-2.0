import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from loguru import logger
from pydantic import BaseModel, Field

from core.detector import YOLOv8FaceDetector
from core.face_quality import FaceQualityAssessor
from core.face_alignment import FaceAligner
from core.schemas import FaceQualityResult


class EnrollmentStatus:
    PASS = "PASS"
    NO_PHOTO = "NO_PHOTO"
    CORRUPT_IMAGE = "CORRUPT_IMAGE"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    LOW_QUALITY = "LOW_QUALITY"
    DUPLICATE_REGISTER_NUMBER = "DUPLICATE_REGISTER_NUMBER"
    DUPLICATE_PHOTO = "DUPLICATE_PHOTO"
    IDENTITY_CONFLICT = "IDENTITY_CONFLICT"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    FAISS_ERROR = "FAISS_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    ERROR = "ERROR"


class FaceValidationResult(BaseModel):
    """
    Standardized result for single student photo validation.
    """
    is_valid: bool = False
    status: str = EnrollmentStatus.ERROR
    rejection_reason: Optional[str] = None
    num_faces_detected: int = 0
    quality_score: float = 0.0
    quality_metrics: Optional[FaceQualityResult] = None
    aligned_tensor: Optional[Any] = Field(default=None, description="Preprocessed 112x112 normalized tensor")


class EnrollmentValidationPipeline:
    """
    Safe Face Validation & Quality Gating Pipeline for Real Student Enrollment.
    Enforces:
    1. Safe decoding of image binaries.
    2. Strict Single-Face Requirement (exactly 1 face; 0 or 2+ rejected).
    3. Phase 5 Quality Gating (sharpness, brightness, face area).
    4. 5-point landmark alignment & ArcFace tensor normalization.
    """

    def __init__(
        self,
        detector: Optional[YOLOv8FaceDetector] = None,
        assessor: Optional[FaceQualityAssessor] = None,
        aligner: Optional[FaceAligner] = None
    ):
        self.detector = detector or YOLOv8FaceDetector()
        self.assessor = assessor or FaceQualityAssessor()
        self.aligner = aligner or FaceAligner()

    def decode_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """Safely decodes image bytes into a BGR numpy array."""
        if not image_data or len(image_data) == 0:
            return None
        try:
            arr = np.frombuffer(image_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None and img.size > 0:
                return img
            return None
        except Exception as e:
            logger.warning(f"Image decode failed: {e}")
            return None

    def validate_photo(
        self,
        image_input: Optional[np.ndarray | bytes],
        student_id: str = "temp_student",
        sample_label: str = "frontal"
    ) -> FaceValidationResult:
        """
        Validates an enrollment photo through detection, single-face check,
        quality assessment, and alignment.
        """
        # 1. Check for missing photo
        if image_input is None:
            return FaceValidationResult(
                is_valid=False,
                status=EnrollmentStatus.NO_PHOTO,
                rejection_reason="No photo provided for student.",
                num_faces_detected=0
            )

        # 2. Decode image if bytes
        if isinstance(image_input, bytes):
            img = self.decode_image(image_input)
            if img is None:
                return FaceValidationResult(
                    is_valid=False,
                    status=EnrollmentStatus.CORRUPT_IMAGE,
                    rejection_reason="Image file is corrupt, empty, or unsupported format.",
                    num_faces_detected=0
                )
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                return FaceValidationResult(
                    is_valid=False,
                    status=EnrollmentStatus.CORRUPT_IMAGE,
                    rejection_reason="Empty image array provided.",
                    num_faces_detected=0
                )
            img = image_input
        else:
            return FaceValidationResult(
                is_valid=False,
                status=EnrollmentStatus.CORRUPT_IMAGE,
                rejection_reason=f"Unsupported image input type: {type(image_input)}",
                num_faces_detected=0
            )

        # 3. Detect Faces
        if hasattr(self.detector, "detect_faces"):
            detections = self.detector.detect_faces(img)
        elif hasattr(self.detector, "detect"):
            detections = self.detector.detect(img)
        else:
            detections = self.detector(img)
        num_faces = len(detections)

        # 4. Strict Single-Face Requirement
        if num_faces == 0:
            return FaceValidationResult(
                is_valid=False,
                status=EnrollmentStatus.NO_FACE,
                rejection_reason="No face detected in enrollment photo. Clear frontal view required.",
                num_faces_detected=0
            )
        elif num_faces > 1:
            return FaceValidationResult(
                is_valid=False,
                status=EnrollmentStatus.MULTIPLE_FACES,
                rejection_reason=f"Multiple faces detected ({num_faces} faces). Enrollment requires exactly ONE student face.",
                num_faces_detected=num_faces
            )

        # Exactly 1 face detected
        det = detections[0]

        # 5. Quality Assessment
        face_id = f"enroll_{student_id}_{sample_label}"
        quality_res = self.assessor.assess(det, img, face_id)

        if quality_res.quality_status != "RECOGNITION_READY":
            return FaceValidationResult(
                is_valid=False,
                status=EnrollmentStatus.LOW_QUALITY,
                rejection_reason=f"Face quality check failed: {quality_res.rejection_reason}",
                num_faces_detected=1,
                quality_metrics=quality_res
            )

        # 6. Alignment & Normalization
        if not quality_res.keypoints or len(quality_res.keypoints) != 5:
            return FaceValidationResult(
                is_valid=False,
                status=EnrollmentStatus.LOW_QUALITY,
                rejection_reason="Face missing required 5 facial landmarks for ArcFace alignment.",
                num_faces_detected=1,
                quality_metrics=quality_res
            )

        try:
            aligned_face = self.aligner.align(img, quality_res.keypoints)
            tensor = self.aligner.normalize(aligned_face)
        except Exception as e:
            return FaceValidationResult(
                is_valid=False,
                status=EnrollmentStatus.ERROR,
                rejection_reason=f"Alignment/normalization failed: {e}",
                num_faces_detected=1,
                quality_metrics=quality_res
            )

        # Calculate quality score (0.0 to 1.0)
        q_score = min(1.0, quality_res.sharpness / 200.0)

        return FaceValidationResult(
            is_valid=True,
            status=EnrollmentStatus.PASS,
            rejection_reason=None,
            num_faces_detected=1,
            quality_score=q_score,
            quality_metrics=quality_res,
            aligned_tensor=tensor
        )

    # Alias for convenience
    validate_image = validate_photo

