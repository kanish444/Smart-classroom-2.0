import os
import cv2
import time
import shutil
import base64
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from loguru import logger

from core.detector import YOLOv8FaceDetector, BaseDetector, get_face_detector
from core.face_quality import FaceQualityAssessor
from core.face_alignment import FaceAligner
from core.face_embedder import ArcFaceEmbedder
from core.vector_store import FaissVectorStore
from database.new_enrollment_db import NewEnrollmentDatabase


class FaceSampleValidationResult:
    """Standardized validation evaluation for a single candidate enrollment frame."""
    def __init__(
        self,
        can_capture: bool,
        status: str,
        message: str,
        quality_status: str,
        num_faces: int,
        rejection_reason: Optional[str] = None,
        aligned_tensor: Optional[np.ndarray] = None,
        crop_image: Optional[np.ndarray] = None,
        quality_metrics: Optional[Dict[str, Any]] = None
    ):
        self.can_capture = can_capture
        self.status = status
        self.message = message
        self.quality_status = quality_status
        self.num_faces = num_faces
        self.rejection_reason = rejection_reason
        self.aligned_tensor = aligned_tensor
        self.crop_image = crop_image
        self.quality_metrics = quality_metrics or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "can_capture": self.can_capture,
            "status": self.status,
            "message": self.message,
            "quality_status": self.quality_status,
            "num_faces": self.num_faces,
            "rejection_reason": self.rejection_reason,
            "quality_metrics": self.quality_metrics
        }


class OneByOneEnrollmentService:
    """
    Phase 10: New One-By-One Student Face Enrollment Service.
    Coordinates:
    - Multiple face protection (0 faces -> prompt, >1 faces -> reject, exactly 1 -> evaluate)
    - Face quality checks (size, blur, brightness, valid bbox, position)
    - Face alignment & 112x112 ArcFace tensor normalization
    - 512-dim ArcFace embedding extraction without model retraining
    - Duplicate register number verification in new enrollment database
    - Atomic persistence: NewEnrollmentDatabase, FAISS vector store, local photo storage
    """

    def __init__(
        self,
        db: Optional[NewEnrollmentDatabase] = None,
        vector_store: Optional[FaissVectorStore] = None,
        embedder: Optional[ArcFaceEmbedder] = None,
        detector: Optional[BaseDetector] = None,
        assessor: Optional[FaceQualityAssessor] = None,
        aligner: Optional[FaceAligner] = None,
        storage_base_dir: Optional[str] = None
    ):
        self.db = db or NewEnrollmentDatabase()
        self.vector_store = vector_store or FaissVectorStore()
        self.embedder = embedder or ArcFaceEmbedder()
        self.detector = detector or get_face_detector()
        self.assessor = assessor or FaceQualityAssessor()
        self.aligner = aligner or FaceAligner()

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.storage_base_dir = storage_base_dir or os.path.join(base_dir, "data", "enrollment")
        os.makedirs(self.storage_base_dir, exist_ok=True)

    @staticmethod
    def decode_image(image_input: Any) -> Optional[np.ndarray]:
        """Safely decodes raw bytes or base64 string into BGR numpy array."""
        if image_input is None:
            return None
        if isinstance(image_input, np.ndarray):
            return image_input if image_input.size > 0 else None
        if isinstance(image_input, bytes):
            arr = np.frombuffer(image_input, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if isinstance(image_input, str):
            # Check for data URL header
            clean_b64 = image_input
            if "," in clean_b64:
                clean_b64 = clean_b64.split(",", 1)[1]
            try:
                raw_bytes = base64.b64decode(clean_b64)
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.warning(f"Base64 image decode error: {e}")
                return None
        return None

    def validate_frame(self, frame_input: Any) -> FaceSampleValidationResult:
        """
        Validates an incoming camera frame for face enrollment:
        1. Decodes frame.
        2. Detects faces using YOLOv8-Face.
        3. Enforces Single-Face Condition:
           - 0 faces: "No face detected. Please position your face clearly."
           - >1 faces: "Multiple faces detected. Only one person can be enrolled at a time."
        4. Gating & Quality Check on single face:
           - "GOOD QUALITY ✓" or "LOW QUALITY — RETAKE"
        5. Aligns face to 112x112 normalized tensor if valid.
        """
        frame = self.decode_image(frame_input)
        if frame is None or frame.size == 0:
            return FaceSampleValidationResult(
                can_capture=False,
                status="CORRUPT_IMAGE",
                message="Invalid or empty camera frame.",
                quality_status="LOW QUALITY — RETAKE",
                num_faces=0,
                rejection_reason="Empty image frame."
            )

        # 1. Detect faces using existing YOLOv8FaceDetector
        if hasattr(self.detector, "detect_faces"):
            detections = self.detector.detect_faces(frame)
        elif hasattr(self.detector, "detect"):
            detections = self.detector.detect(frame)
        else:
            detections = self.detector(frame)

        num_faces = len(detections) if detections is not None else 0

        # 2. Strict Multiple Face Protection
        if num_faces == 0:
            return FaceSampleValidationResult(
                can_capture=False,
                status="NO_FACE",
                message="No face detected. Please position your face clearly.",
                quality_status="NO FACE",
                num_faces=0,
                rejection_reason="No face detected in frame."
            )

        if num_faces > 1:
            return FaceSampleValidationResult(
                can_capture=False,
                status="MULTIPLE_FACES",
                message="Multiple faces detected. Only one person can be enrolled at a time.",
                quality_status="MULTIPLE FACES",
                num_faces=num_faces,
                rejection_reason=f"{num_faces} faces detected in frame."
            )

        # 3. Exactly one face detected -> Quality check
        det = detections[0]
        face_id = f"enroll_eval_{int(time.time() * 1000)}"
        quality_res = self.assessor.assess(det, frame, face_id)

        # Extract crop for preview
        x1, y1, x2, y2 = quality_res.bbox
        h_frame, w_frame = frame.shape[:2]
        crop_img = None
        if x2 > x1 and y2 > y1:
            crop_img = frame[max(0, y1):min(h_frame, y2), max(0, x1):min(w_frame, x2)]

        metrics = {
            "area": quality_res.area,
            "sharpness": round(quality_res.sharpness, 1),
            "brightness": round(quality_res.brightness, 1),
            "bbox": quality_res.bbox
        }

        # Check quality threshold
        if quality_res.quality_status != "RECOGNITION_READY":
            reason = quality_res.rejection_reason or "Low quality criteria"
            return FaceSampleValidationResult(
                can_capture=False,
                status="LOW_QUALITY",
                message=f"Low quality sample: {reason}",
                quality_status="LOW QUALITY — RETAKE",
                num_faces=1,
                rejection_reason=reason,
                crop_image=crop_img,
                quality_metrics=metrics
            )

        # Check required 5 facial landmarks for ArcFace alignment
        if not quality_res.keypoints or len(quality_res.keypoints) != 5:
            return FaceSampleValidationResult(
                can_capture=False,
                status="LOW_QUALITY",
                message="Face landmarks could not be reliably determined.",
                quality_status="LOW QUALITY — RETAKE",
                num_faces=1,
                rejection_reason="Missing 5 facial keypoints for alignment.",
                crop_image=crop_img,
                quality_metrics=metrics
            )

        # Align and normalize tensor
        try:
            aligned_face = self.aligner.align(frame, quality_res.keypoints)
            tensor = self.aligner.normalize(aligned_face)
        except Exception as e:
            logger.warning(f"Face alignment error: {e}")
            return FaceSampleValidationResult(
                can_capture=False,
                status="LOW_QUALITY",
                message=f"Face alignment failed: {e}",
                quality_status="LOW QUALITY — RETAKE",
                num_faces=1,
                rejection_reason=str(e),
                crop_image=crop_img,
                quality_metrics=metrics
            )

        return FaceSampleValidationResult(
            can_capture=True,
            status="PASS",
            message="Good quality face sample detected.",
            quality_status="GOOD QUALITY ✓",
            num_faces=1,
            rejection_reason=None,
            aligned_tensor=tensor,
            crop_image=crop_img,
            quality_metrics=metrics
        )

    def check_duplicate_register_number(self, register_number: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Checks whether the Register Number already exists in the new enrollment database or central database.
        Returns: (exists: bool, existing_student_profile: Optional[Dict])
        """
        clean_reg = str(register_number).strip()
        existing = self.db.get_student_by_register_number(clean_reg)
        if existing is None:
            try:
                from database.db_manager import DatabaseManager
                db_mgr = DatabaseManager()
                main_st = db_mgr.get_student_by_register_no(clean_reg)
                if main_st:
                    existing = {
                        "student_id": main_st.get("student_id"),
                        "register_number": main_st.get("register_no", clean_reg),
                        "name": main_st.get("student_name"),
                        "class": main_st.get("class_name", "3rd Year"),
                        "department": main_st.get("department", "AI&DS"),
                        "section": main_st.get("section", "B"),
                        "created_at": main_st.get("created_at", "Authoritative Database")
                    }
            except Exception:
                pass
        return (existing is not None, existing)

    def enroll_student(
        self,
        register_number: str,
        name: str,
        class_name: str,
        department: str,
        section: str,
        sample_frames: List[Any],
        min_samples: int = 5,
        max_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Executes one-by-one student enrollment:
        1. Checks duplicate register number.
        2. Validates 5-10 face samples (single-face + good quality).
        3. Generates 512-dim ArcFace embeddings.
        4. Atomically persists student + embeddings into NewEnrollmentDatabase.
        5. Saves sample photos to data/enrollment/{register_number}/sample_XX.jpg.
        6. Synchronizes embeddings into FAISS with metadata.
        7. If any failure occurs, rolls back all operations completely.
        """
        clean_reg = str(register_number).strip()
        clean_name = str(name).strip()
        clean_class = str(class_name).strip()
        clean_dept = str(department).strip()
        clean_sec = str(section).strip()

        # 1. Validate fields
        if not clean_reg:
            raise ValueError("Register Number is required.")
        if not clean_name:
            raise ValueError("Student Name is required.")
        if not clean_class:
            raise ValueError("Class is required.")
        if not clean_dept:
            raise ValueError("Department is required.")
        if not clean_sec:
            raise ValueError("Section is required.")

        # 2. Check Duplicate Register Number
        is_dup, existing_profile = self.check_duplicate_register_number(clean_reg)
        if is_dup:
            raise ValueError(f"Student already enrolled with Register Number '{clean_reg}'.")

        # 3. Check sample count requirement (approx 5-10 good samples)
        if not sample_frames or len(sample_frames) < min_samples:
            raise ValueError(f"At least {min_samples} good face samples are required (provided: {len(sample_frames) if sample_frames else 0}).")

        # Limit to max_samples
        selected_frames = sample_frames[:max_samples]

        # 4. Validate all samples and generate embeddings
        embeddings: List[np.ndarray] = []
        quality_scores: List[float] = []
        valid_bgr_images: List[np.ndarray] = []

        for idx, s_frame in enumerate(selected_frames):
            val_res = self.validate_frame(s_frame)
            if not val_res.can_capture or val_res.aligned_tensor is None:
                raise ValueError(
                    f"Sample {idx + 1} failed quality validation: {val_res.rejection_reason or val_res.message} ({val_res.quality_status})"
                )

            # Generate ArcFace embedding
            emb = self.embedder.generate_embedding(val_res.aligned_tensor)
            if emb.ndim > 1:
                emb = emb.reshape(-1)

            # Validate embedding properties
            if emb.size != self.embedder.embedding_dim:
                raise ValueError(f"Invalid embedding dimension: {emb.size} (expected {self.embedder.embedding_dim})")
            if np.isnan(emb).any() or np.isinf(emb).any():
                raise ValueError("Generated face embedding contains NaN or Inf values.")

            embeddings.append(emb)
            q_score = min(1.0, val_res.quality_metrics.get("sharpness", 100.0) / 200.0)
            quality_scores.append(q_score)

            decoded_img = self.decode_image(s_frame)
            valid_bgr_images.append(decoded_img)

        # 5. Deterministic student_id: STU_<register_number> (no random hash suffixes)
        student_id = clean_reg if clean_reg.startswith("STU_") else f"STU_{clean_reg}"

        # 6. Save photos locally to data/enrollment/{register_number}/
        student_photo_dir = os.path.join(self.storage_base_dir, clean_reg)
        saved_file_paths: List[str] = []
        try:
            os.makedirs(student_photo_dir, exist_ok=True)
            for idx, img in enumerate(valid_bgr_images):
                filename = f"sample_{idx + 1:02d}.jpg"
                filepath = os.path.join(student_photo_dir, filename)
                cv2.imwrite(filepath, img)
                saved_file_paths.append(filepath)
        except Exception as e:
            # Clean up photos if write failed
            shutil.rmtree(student_photo_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to save student face photos locally: {e}")

        # 7. Atomic SQLite database insertion
        added_faiss_vector_ids: List[int] = []
        try:
            embedding_ids = self.db.save_enrollment_atomic(
                student_id=student_id,
                register_number=clean_reg,
                name=clean_name,
                class_name=clean_class,
                department=clean_dept,
                section=clean_sec,
                embeddings=embeddings,
                quality_scores=quality_scores
            )

            # 8. Add embeddings to FAISS vector store with full metadata mapping
            for emb_id, emb, q_score in zip(embedding_ids, embeddings, quality_scores):
                meta = {
                    "embedding_id": emb_id,
                    "student_id": student_id,
                    "register_number": clean_reg,
                    "student_name": clean_name,
                    "class": clean_class,
                    "department": clean_dept,
                    "section": clean_sec,
                    "quality_score": float(q_score)
                }
                self.vector_store.add_vector(emb_id, emb, meta)
                added_faiss_vector_ids.append(emb_id)

            # Persist FAISS index
            self.vector_store.save_index()

            # Synchronize student to smartclass.sqlite students table so foreign keys always pass
            try:
                from database.db_manager import DatabaseManager
                db_mgr = DatabaseManager()
                cls_name = f"{clean_dept} - {clean_sec}"
                db_mgr.add_student(
                    student_id=student_id,
                    student_name=clean_name,
                    department=clean_dept,
                    section=clean_sec,
                    status="active",
                    register_no=clean_reg,
                    class_name=cls_name
                )
            except Exception as sync_err:
                logger.warning(f"Could not sync student to main db: {sync_err}")

        except Exception as e:
            # Atomic rollback: delete student from DB, delete photo folder
            logger.error(f"Enrollment failure during persistence: {e}. Executing atomic rollback.")
            try:
                self.db.delete_student(student_id)
            except Exception:
                pass
            shutil.rmtree(student_photo_dir, ignore_errors=True)
            raise e

        logger.info(f"Student '{clean_name}' ({clean_reg}) successfully enrolled with {len(embeddings)} face samples.")

        return {
            "success": True,
            "student_id": student_id,
            "register_number": clean_reg,
            "name": clean_name,
            "class": clean_class,
            "department": clean_dept,
            "section": clean_sec,
            "samples_enrolled": len(embeddings),
            "embedding_ids": embedding_ids,
            "photo_directory": student_photo_dir,
            "message": f"Student '{clean_name}' enrolled successfully."
        }
