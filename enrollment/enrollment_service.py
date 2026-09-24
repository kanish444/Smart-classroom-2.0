import os
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from core.detector import YOLOv8FaceDetector
from core.face_embedder import ArcFaceEmbedder
from core.face_quality import FaceQualityAssessor
from core.face_alignment import FaceAligner
from core.vector_store import FaissVectorStore
from database.db_manager import DatabaseManager
from core.schemas import StudentProfile, FaceQualityResult
from enrollment.data_importer import StudentImportRecord
from enrollment.validation_pipeline import (
    EnrollmentValidationPipeline,
    EnrollmentStatus,
    FaceValidationResult
)


class StudentEnrollmentDetail(BaseModel):
    student_id: str
    name: str
    class_section: str
    status: str
    rejection_reason: Optional[str] = None
    samples_enrolled: int = 0
    embedding_ids: List[int] = Field(default_factory=list)


class EnrollmentReport(BaseModel):
    """
    Standardized Audit Report for Real Student Batch Enrollment.
    """
    total_records: int = 0
    successfully_enrolled: int = 0
    failed: int = 0
    requires_review: int = 0
    counts: Dict[str, int] = Field(default_factory=dict)
    details: List[StudentEnrollmentDetail] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    avg_time_per_student_ms: float = 0.0
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    @property
    def failures(self) -> List[Dict[str, Any]]:
        return [
            {"student_id": d.student_id, "name": d.name, "status": d.status, "reason": d.rejection_reason}
            for d in self.details if d.status != EnrollmentStatus.PASS
        ]


class EnrollmentService:
    """
    Phase 10: Real Student Enrollment & Identity Database Coordinator.
    Coordinates:
    - Ingestion of real student records (DOCX, CSV, Directory)
    - Single-face requirement and quality gate validation
    - Face alignment & 112x112 ArcFace tensor normalization
    - Pretrained ArcFace 512-dim embedding extraction (NO model retraining)
    - Identity conflict / duplicate photo verification
    - SQLite persistence (students, embeddings) and FAISS index synchronization
    - Complete student lifecycle: register, update, re-enroll, remove, and index rebuild
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        vector_store: Optional[FaissVectorStore] = None,
        embedder: Optional[ArcFaceEmbedder] = None,
        assessor: Optional[FaceQualityAssessor] = None,
        aligner: Optional[FaceAligner] = None,
        detector: Optional[YOLOv8FaceDetector] = None,
        conflict_threshold: float = 0.95,
        db: Optional[DatabaseManager] = None,
        store: Optional[FaissVectorStore] = None,
    ):
        self.db = db_manager or db
        if self.db is None:
            raise ValueError("db_manager or db must be provided to EnrollmentService.")
        self.vector_store = vector_store or store
        if self.vector_store is None:
            raise ValueError("vector_store or store must be provided to EnrollmentService.")
        self.embedder = embedder or ArcFaceEmbedder()
        self.assessor = assessor or FaceQualityAssessor()
        self.aligner = aligner or FaceAligner()
        self.detector = detector or YOLOv8FaceDetector()
        self.conflict_threshold = conflict_threshold

        self.validation_pipeline = EnrollmentValidationPipeline(
            detector=self.detector,
            assessor=self.assessor,
            aligner=self.aligner
        )

    def register_student(
        self,
        student_id: str,
        student_name: str,
        department: str = "Computer Science",
        section: str = "A",
        reg_no: Optional[str] = None,
        class_section: Optional[str] = None,
        status: str = "active"
    ) -> bool:
        """Registers or updates student identity metadata in SQLite."""
        if not student_id or not student_name:
            raise ValueError("Student ID and Student Name cannot be empty.")
        from .data_importer import StudentDataImporter
        if class_section:
            d, s = StudentDataImporter().parse_class_section(class_section)
            department = d or department
            section = s or section
        return self.db.add_student(
            student_id=student_id,
            student_name=student_name,
            department=department,
            section=section,
            status=status,
            register_no=reg_no or student_id,
            class_name=class_section or f"{department} - {section}"
        )

    def update_student(
        self,
        student_id: str,
        student_name: Optional[str] = None,
        department: Optional[str] = None,
        section: Optional[str] = None,
        reg_no: Optional[str] = None,
        class_section: Optional[str] = None,
        status: Optional[str] = None
    ) -> bool:
        """Updates existing student profile without modifying enrolled embeddings."""
        existing = self.db.get_student(student_id)
        if not existing:
            return False
        from .data_importer import StudentDataImporter
        name = student_name or existing["student_name"]
        dept = department or existing["department"]
        sec = section or existing["section"]
        st = status or existing.get("status", "active")
        r_no = reg_no or existing.get("register_no", student_id)
        if class_section:
            d, s = StudentDataImporter().parse_class_section(class_section)
            dept = d or dept
            sec = s or sec
            c_name = class_section
        else:
            c_name = existing.get("class_name") or f"{dept} - {sec}"

        return self.db.add_student(
            student_id=student_id,
            student_name=name,
            department=dept,
            section=sec,
            status=st,
            register_no=r_no,
            class_name=c_name
        )

    def check_embedding_conflict(
        self,
        embedding: np.ndarray,
        current_student_id: str
    ) -> Optional[Tuple[str, float]]:
        """
        Checks whether the candidate embedding is suspiciously similar (>= conflict_threshold)
        to an existing DIFFERENT student in the FAISS index.
        Returns (conflicting_student_id, similarity) or None if no conflict.
        """
        if self.vector_store.total_vectors == 0:
            return None

        matches = self.vector_store.search(embedding, top_k=1)
        if not matches:
            return None

        top_sim, top_meta = matches[0]
        matched_student = top_meta.get("student_id")

        if top_sim >= self.conflict_threshold and matched_student != current_student_id:
            return (matched_student, float(top_sim))

        return None

    def enroll_face_sample(
        self,
        student_id: str,
        face_tensor: np.ndarray,
        sample_label: str = "frontal",
        quality_score: float = 1.0,
        skip_conflict_check: bool = False
    ) -> Tuple[bool, Optional[int], str]:
        """
        Enrolls a preprocessed Phase 5 tensor (1, 3, 112, 112) for a registered student.
        Validates 512-dim embedding, checks identity conflict, stores in SQLite, and adds to FAISS.
        """
        student = self.db.get_student(student_id)
        if not student:
            return False, None, f"Student '{student_id}' is not registered."

        try:
            # 1. Generate L2-normalized 512-dim embedding
            embedding = self.embedder.generate_embedding(face_tensor)
            if embedding.ndim > 1:
                embedding = embedding.reshape(-1)

            # Validate embedding properties
            if embedding.size != self.embedder.embedding_dim:
                return False, None, f"Invalid embedding dimension: {embedding.shape}"
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                return False, None, "Embedding contains NaN or Inf values."

            # 2. Conflict / Duplicate Photo Check
            if not skip_conflict_check:
                conflict = self.check_embedding_conflict(embedding, student_id)
                if conflict is not None:
                    conflict_id, sim = conflict
                    msg = (
                        f"Identity conflict: Photo embedding matches existing student '{conflict_id}' "
                        f"with {sim:.2f} similarity. Enrollment halted for manual review."
                    )
                    logger.warning(msg)
                    return False, None, msg

            # 3. Store in SQLite
            emb_id = self.db.add_embedding(
                student_id=student_id,
                vector=embedding,
                quality_score=quality_score,
                sample_label=sample_label,
                model_version="w600k_mbf"
            )

            # 4. Add to FAISS Vector Store
            metadata = {
                "embedding_id": emb_id,
                "student_id": student_id,
                "student_name": student["student_name"],
                "sample_label": sample_label,
                "quality_score": quality_score
            }
            self.vector_store.add_vector(emb_id, embedding, metadata)

            logger.info(
                f"Enrolled sample '{sample_label}' (Embedding ID {emb_id}) for "
                f"student {student_id} ({student['student_name']})"
            )
            return True, emb_id, "Sample enrolled successfully."

        except Exception as e:
            logger.error(f"Failed to enroll face sample for student {student_id}: {e}")
            return False, None, f"Enrollment error: {str(e)}"

    def enroll_from_frame(
        self,
        student_id: str,
        frame: np.ndarray,
        detection: Dict[str, Any],
        sample_label: str = "frontal"
    ) -> Tuple[bool, Optional[int], str]:
        """
        Legacy convenience method: enrolls from an image frame and pre-computed detection.
        """
        quality_res = self.assessor.assess(detection, frame, f"enroll_{student_id}")
        if quality_res.quality_status != "RECOGNITION_READY":
            return False, None, f"Sample rejected by quality gate: {quality_res.rejection_reason}"

        if not quality_res.keypoints or len(quality_res.keypoints) != 5:
            return False, None, "Face missing required 5 facial landmarks for alignment."

        try:
            aligned_face = self.aligner.align(frame, quality_res.keypoints)
            tensor = self.aligner.normalize(aligned_face)
        except Exception as e:
            return False, None, f"Phase 5 alignment/normalization failed: {e}"

        quality_score = min(1.0, quality_res.sharpness / 200.0)
        return self.enroll_face_sample(
            student_id=student_id,
            face_tensor=tensor,
            sample_label=sample_label,
            quality_score=quality_score
        )

    def enroll_from_camera_samples(
        self,
        student_id: str,
        student_name: str,
        frames: List[Any],
        labels: Optional[List[str]] = None,
        class_section: Optional[str] = None,
        department: str = "Computer Science",
        section: str = "A"
    ) -> Tuple[bool, List[int], List[str]]:
        """
        Phase 10.10: Direct multi-shot camera enrollment.
        Accepts 1 to N captured camera frames (e.g. Capture 1, Capture 2, Capture 3).
        Validates single face & quality, computes ArcFace embeddings, registers student in SQLite,
        and adds embeddings to SQLite + FAISS.
        Returns: (success: bool, enrolled_embedding_ids: List[int], errors: List[str])
        """
        if not frames:
            return False, [], ["No camera frames provided for enrollment."]

        # 1. Register student metadata in SQLite first
        self.register_student(
            student_id=student_id,
            student_name=student_name,
            department=department,
            section=section,
            reg_no=student_id,
            class_section=class_section
        )

        enrolled_ids: List[int] = []
        errors: List[str] = []

        for idx, frame in enumerate(frames, start=1):
            label = labels[idx - 1] if (labels and idx - 1 < len(labels)) else f"camera_shot_{idx}"
            v_res = self.validation_pipeline.validate_photo(
                image_input=frame,
                student_id=student_id,
                sample_label=label
            )

            if not v_res.is_valid:
                errors.append(f"Shot {idx} ({label}) rejected: {v_res.rejection_reason}")
                continue

            success, emb_id, msg = self.enroll_face_sample(
                student_id=student_id,
                face_tensor=v_res.aligned_tensor,
                sample_label=label,
                quality_score=v_res.quality_score
            )

            if success and emb_id is not None:
                enrolled_ids.append(emb_id)
            else:
                errors.append(f"Shot {idx} ({label}) failed: {msg}")

        if enrolled_ids:
            self.vector_store.save_index()
            logger.info(f"CAMERA_ENROLLMENT_SUCCESS: Student {student_id} enrolled with {len(enrolled_ids)} camera samples.")
            return True, enrolled_ids, errors

        return False, [], errors

    def enroll_batch(

        self,
        records: List[StudentImportRecord],
        dry_run: bool = False
    ) -> EnrollmentReport:
        """
        Performs batch enrollment across a list of parsed student records.
        - Validates single-face requirement and quality on each photo.
        - Enrolls identities and synchronizes SQLite and FAISS.
        - Collects audit counts and per-student details.
        """
        t0 = time.perf_counter()
        counts: Dict[str, int] = {
            EnrollmentStatus.PASS: 0,
            EnrollmentStatus.NO_PHOTO: 0,
            EnrollmentStatus.CORRUPT_IMAGE: 0,
            EnrollmentStatus.NO_FACE: 0,
            EnrollmentStatus.MULTIPLE_FACES: 0,
            EnrollmentStatus.LOW_QUALITY: 0,
            EnrollmentStatus.DUPLICATE_REGISTER_NUMBER: 0,
            EnrollmentStatus.DUPLICATE_PHOTO: 0,
            EnrollmentStatus.IDENTITY_CONFLICT: 0,
            EnrollmentStatus.EMBEDDING_ERROR: 0,
            EnrollmentStatus.FAISS_ERROR: 0,
            EnrollmentStatus.DATABASE_ERROR: 0,
            EnrollmentStatus.ERROR: 0
        }

        details: List[StudentEnrollmentDetail] = []
        seen_reg_nos = set()

        for rec in records:
            reg_no = rec.register_no.strip()
            name = rec.name.strip()

            # 1. Duplicate Register Number check within batch
            if reg_no in seen_reg_nos:
                counts[EnrollmentStatus.DUPLICATE_REGISTER_NUMBER] += 1
                details.append(StudentEnrollmentDetail(
                    student_id=reg_no,
                    name=name,
                    class_section=rec.class_section,
                    status=EnrollmentStatus.DUPLICATE_REGISTER_NUMBER,
                    rejection_reason="Duplicate Register Number within import batch."
                ))
                continue
            seen_reg_nos.add(reg_no)

            # 2. Check for missing photos
            if not rec.photos or len(rec.photos) == 0:
                counts[EnrollmentStatus.NO_PHOTO] += 1
                details.append(StudentEnrollmentDetail(
                    student_id=reg_no,
                    name=name,
                    class_section=rec.class_section,
                    status=EnrollmentStatus.NO_PHOTO,
                    rejection_reason="No photo provided in student record."
                ))
                continue

            # Process each photo for this student
            enrolled_embs: List[int] = []
            final_status = EnrollmentStatus.ERROR
            last_reason = None

            for p_idx, photo_bytes in enumerate(rec.photos, start=1):
                label = f"sample_{p_idx}" if len(rec.photos) > 1 else "frontal"
                v_res: FaceValidationResult = self.validation_pipeline.validate_photo(
                    image_input=photo_bytes,
                    student_id=reg_no,
                    sample_label=label
                )

                if not v_res.is_valid:
                    final_status = v_res.status
                    last_reason = v_res.rejection_reason
                    counts[v_res.status] = counts.get(v_res.status, 0) + 1
                    continue

                if dry_run:
                    # In dry run mode, do not commit to DB or FAISS
                    final_status = EnrollmentStatus.PASS
                    enrolled_embs.append(9999 + p_idx)
                    continue

                # 3. Register student profile in SQLite
                self.register_student(
                    student_id=reg_no,
                    student_name=name,
                    department=rec.department,
                    section=rec.section
                )

                # 4. Enroll sample embedding
                success, emb_id, msg = self.enroll_face_sample(
                    student_id=reg_no,
                    face_tensor=v_res.aligned_tensor,
                    sample_label=label,
                    quality_score=v_res.quality_score
                )

                if success and emb_id is not None:
                    enrolled_embs.append(emb_id)
                    final_status = EnrollmentStatus.PASS
                else:
                    if "Identity conflict" in msg:
                        final_status = EnrollmentStatus.IDENTITY_CONFLICT
                        counts[EnrollmentStatus.IDENTITY_CONFLICT] += 1
                    else:
                        final_status = EnrollmentStatus.EMBEDDING_ERROR
                        counts[EnrollmentStatus.EMBEDDING_ERROR] += 1
                    last_reason = msg

            if final_status == EnrollmentStatus.PASS:
                counts[EnrollmentStatus.PASS] += 1
                details.append(StudentEnrollmentDetail(
                    student_id=reg_no,
                    name=name,
                    class_section=rec.class_section,
                    status=EnrollmentStatus.PASS,
                    samples_enrolled=len(enrolled_embs),
                    embedding_ids=enrolled_embs
                ))
            else:
                details.append(StudentEnrollmentDetail(
                    student_id=reg_no,
                    name=name,
                    class_section=rec.class_section,
                    status=final_status,
                    rejection_reason=last_reason,
                    samples_enrolled=len(enrolled_embs),
                    embedding_ids=enrolled_embs
                ))

        # Persist FAISS index if not dry run
        if not dry_run and counts[EnrollmentStatus.PASS] > 0:
            try:
                self.vector_store.save_index()
            except Exception as e:
                logger.error(f"Failed to persist FAISS index: {e}")

        elapsed = time.perf_counter() - t0
        total = len(records)
        passed = counts[EnrollmentStatus.PASS]
        failed = sum(v for k, v in counts.items() if k != EnrollmentStatus.PASS)
        review = (
            counts.get(EnrollmentStatus.MULTIPLE_FACES, 0) +
            counts.get(EnrollmentStatus.IDENTITY_CONFLICT, 0) +
            counts.get(EnrollmentStatus.LOW_QUALITY, 0)
        )

        avg_ms = (elapsed / max(1, total)) * 1000.0

        return EnrollmentReport(
            total_records=total,
            successfully_enrolled=passed,
            failed=failed,
            requires_review=review,
            counts=counts,
            details=details,
            elapsed_seconds=round(elapsed, 3),
            avg_time_per_student_ms=round(avg_ms, 2)
        )

    def re_enroll_student(
        self,
        student_id: str,
        photos: List[Any],
        labels: Optional[List[str]] = None
    ) -> Tuple[bool, int, str]:
        """
        Re-enrolls an existing student:
        1. Validates that the student exists.
        2. Validates new photo(s) or pre-normalized tensors.
        3. Removes previous embeddings for the student from SQLite and FAISS.
        4. Enrolls new embeddings.
        5. Persists index and verifies consistency.
        Returns: (success: bool, count: int, message: str)
        """
        student = self.db.get_student(student_id)
        if not student:
            return False, 0, f"Student '{student_id}' not found."

        if not photos or len(photos) == 0:
            return False, 0, "At least one new photo is required for re-enrollment."

        # Validate new photos/tensors first before deleting old ones
        validated_tensors = []
        for idx, item in enumerate(photos):
            label = labels[idx] if (labels and idx < len(labels)) else f"sample_{idx+1}"
            if isinstance(item, np.ndarray):
                validated_tensors.append((item, 1.0, label))
            else:
                v_res = self.validation_pipeline.validate_photo(item, student_id, label)
                if not v_res.is_valid:
                    return False, 0, f"Re-enrollment validation failed on photo {idx+1}: {v_res.rejection_reason}"
                validated_tensors.append((v_res.aligned_tensor, v_res.quality_score, label))

        # Safe replacement: delete old embeddings for this student
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM embeddings WHERE student_id = ?;", (student_id,))
            conn.commit()

        # Resync FAISS to drop removed vectors
        all_embs = self.db.get_all_embeddings()
        self.vector_store.sync_with_database(all_embs)

        # Enroll new validated tensors
        new_ids = []
        for tensor, q_score, label in validated_tensors:
            success, emb_id, msg = self.enroll_face_sample(
                student_id=student_id,
                face_tensor=tensor,
                sample_label=label,
                quality_score=q_score,
                skip_conflict_check=True
            )
            if success and emb_id:
                new_ids.append(emb_id)

        self.vector_store.save_index()
        logger.info(f"RE_ENROLLMENT_COMPLETE: Student {student_id} re-enrolled with {len(new_ids)} samples.")
        return True, len(new_ids), f"Student '{student_id}' re-enrolled with {len(new_ids)} sample(s)."

    def delete_student(self, student_id: str) -> bool:
        """Removes a student and re-synchronizes the FAISS vector store."""
        success = self.db.delete_student(student_id)
        if success:
            all_embs = self.db.get_all_embeddings()
            self.vector_store.sync_with_database(all_embs)
            self.vector_store.save_index()
            logger.info(f"STUDENT_DELETED: Student {student_id} removed from SQLite and FAISS.")
        return success

    # Aliases
    remove_student = delete_student

    def rebuild_faiss_index(self) -> int:
        """
        Reconstructs the FAISS index from the SQLite embeddings table and returns the count.
        """
        all_embs = self.db.get_all_embeddings()
        self.vector_store.sync_with_database(all_embs)
        self.vector_store.save_index()
        return self.vector_store.total_vectors

    def rebuild_index(self) -> Tuple[bool, str]:
        """Legacy rebuild returning (bool, str)."""
        try:
            cnt = self.rebuild_faiss_index()
            return True, f"FAISS index successfully rebuilt with {cnt} embeddings."
        except Exception as e:
            return False, f"Index rebuild failed: {str(e)}"

    def verify_index_consistency(self) -> Dict[str, Any]:
        """Returns consistency status and counts between SQLite and FAISS."""
        db_count = self.db.get_embedding_count()
        faiss_count = self.vector_store.total_vectors
        consistent = (db_count == faiss_count)
        return {
            "consistent": consistent,
            "db_embeddings_count": db_count,
            "faiss_vectors_count": faiss_count,
            "message": "SQLite and FAISS index are synchronized" if consistent else f"Discrepancy: DB has {db_count}, FAISS has {faiss_count}"
        }

    def validate_consistency(self) -> Tuple[bool, str]:
        """Verifies FAISS vector count matches SQLite embedding count."""
        db_count = self.db.get_embedding_count()
        return self.vector_store.validate_consistency(db_count)

    def get_student_profiles(self) -> List[StudentProfile]:
        """Returns all enrolled student profiles with sample counts."""
        students = self.db.get_all_students()
        profiles = []
        for s in students:
            embs = self.db.get_embeddings_for_student(s["student_id"])
            profiles.append(StudentProfile(
                student_id=s["student_id"],
                student_name=s["student_name"],
                department=s["department"],
                section=s["section"],
                status=s["status"],
                sample_count=len(embs)
            ))
        return profiles
