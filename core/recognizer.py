import time
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

from core.face_embedder import ArcFaceEmbedder
from core.vector_store import FaissVectorStore
from core.schemas import RecognitionReadyFace, RecognitionResult, RecognitionStatus, FaceQualityResult
from config.settings import get_settings


class FaceRecognizer:
    """
    Phase 6 Face Recognition & Identity Matching Engine:
    - Receives Phase 5 RecognitionReadyFace tensors
    - Generates 512-dim ArcFace embedding vectors
    - Queries FAISS vector store for nearest enrolled identities
    - Applies strict threshold-based Unknown Rejection
    - Returns standardized, typed RecognitionResult instances
    """

    def __init__(
        self,
        embedder: Optional[ArcFaceEmbedder] = None,
        vector_store: Optional[FaissVectorStore] = None,
        threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None,
        top_k: Optional[int] = None
    ):
        self.settings = get_settings().recognition
        self.embedder = embedder or ArcFaceEmbedder()
        self.vector_store = vector_store or FaissVectorStore()
        self.threshold = threshold if threshold is not None else self.settings.similarity_threshold
        self.margin_threshold = (
            margin_threshold if margin_threshold is not None
            else getattr(self.settings, "margin_threshold", 0.08)
        )
        self.top_k = top_k if top_k is not None else getattr(self.settings, "top_k", 10)
        self.model_name = self.settings.model_name

    def set_threshold(self, new_threshold: float):
        """Allows dynamically updating the recognition threshold for calibration/tuning."""
        if not (0.0 <= new_threshold <= 1.0):
            raise ValueError(f"Similarity threshold must be between 0.0 and 1.0, got {new_threshold}.")
        self.threshold = float(new_threshold)
        logger.info(f"FaceRecognizer threshold updated to {self.threshold:.3f}")

    def set_margin_threshold(self, new_margin: float):
        """Allows dynamically updating the margin threshold."""
        if not (0.0 <= new_margin <= 1.0):
            raise ValueError(f"Margin threshold must be between 0.0 and 1.0, got {new_margin}.")
        self.margin_threshold = float(new_margin)
        logger.info(f"FaceRecognizer margin threshold updated to {self.margin_threshold:.3f}")

    def recognize_face(self, ready_face: RecognitionReadyFace) -> RecognitionResult:
        """
        Recognizes an individual Phase 5 processed face.
        Guarantees safe exception handling: will return structured ERROR or INVALID result
        without crashing the caller.
        """
        t0 = time.perf_counter()
        face_id = "unknown_face"
        bbox = []
        qm: Optional[FaceQualityResult] = None

        if ready_face is not None:
            bbox = ready_face.original_bbox
            if ready_face.quality_metrics:
                qm = ready_face.quality_metrics
                face_id = qm.face_id

        # Validate input readiness
        if ready_face is None or ready_face.aligned_face_tensor is None:
            proc_time_ms = (time.perf_counter() - t0) * 1000.0
            return RecognitionResult(
                face_id=face_id,
                matched_student_id=None,
                matched_student_name=None,
                similarity=0.0,
                status=RecognitionStatus.INVALID,
                threshold=self.threshold,
                embedding_model=self.model_name,
                processing_time_ms=proc_time_ms,
                bbox=bbox,
                quality_metrics=qm,
                second_best_student_id=None,
                second_best_similarity=0.0,
                margin=0.0
            )

        # 1. Generate ArcFace Embedding
        try:
            embedding = self.embedder.generate_embedding(ready_face.aligned_face_tensor)
        except Exception as e:
            logger.error(f"Embedding generation failed for face '{face_id}': {e}")
            proc_time_ms = (time.perf_counter() - t0) * 1000.0
            return RecognitionResult(
                face_id=face_id,
                matched_student_id=None,
                matched_student_name=None,
                similarity=0.0,
                status=RecognitionStatus.ERROR,
                threshold=self.threshold,
                embedding_model=self.model_name,
                processing_time_ms=proc_time_ms,
                bbox=bbox,
                quality_metrics=qm,
                second_best_student_id=None,
                second_best_similarity=0.0,
                margin=0.0
            )

        # 2. Query FAISS Vector Store for Top-K Candidates
        try:
            matches = self.vector_store.search(embedding, top_k=self.top_k)
        except Exception as e:
            logger.error(f"FAISS search failed for face '{face_id}': {e}")
            proc_time_ms = (time.perf_counter() - t0) * 1000.0
            return RecognitionResult(
                face_id=face_id,
                matched_student_id=None,
                matched_student_name=None,
                similarity=0.0,
                status=RecognitionStatus.ERROR,
                threshold=self.threshold,
                embedding_model=self.model_name,
                processing_time_ms=proc_time_ms,
                bbox=bbox,
                quality_metrics=qm,
                second_best_student_id=None,
                second_best_similarity=0.0,
                margin=0.0
            )

        proc_time_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Handle Empty Enrollment Database / Vector Store
        if not matches:
            return RecognitionResult(
                face_id=face_id,
                matched_student_id=None,
                matched_student_name=None,
                similarity=0.0,
                status=RecognitionStatus.UNKNOWN,
                threshold=self.threshold,
                embedding_model=self.model_name,
                processing_time_ms=proc_time_ms,
                bbox=bbox,
                quality_metrics=qm,
                second_best_student_id=None,
                second_best_similarity=0.0,
                margin=0.0
            )

        # 4. Multi-Template Candidate Aggregation by Student ID
        student_candidates: Dict[str, Dict[str, Any]] = {}
        for sim, meta in matches:
            sid = meta.get("student_id")
            if not sid:
                continue
            sname = meta.get("student_name", "Unknown")
            if sid not in student_candidates or sim > student_candidates[sid]["similarity"]:
                student_candidates[sid] = {
                    "student_id": sid,
                    "student_name": sname,
                    "similarity": float(sim)
                }

        if not student_candidates:
            return RecognitionResult(
                face_id=face_id,
                matched_student_id=None,
                matched_student_name=None,
                similarity=0.0,
                status=RecognitionStatus.UNKNOWN,
                threshold=self.threshold,
                embedding_model=self.model_name,
                processing_time_ms=proc_time_ms,
                bbox=bbox,
                quality_metrics=qm,
                second_best_student_id=None,
                second_best_similarity=0.0,
                margin=0.0
            )

        ranked = sorted(student_candidates.values(), key=lambda c: c["similarity"], reverse=True)
        best = ranked[0]
        best_id = best["student_id"]
        best_name = best["student_name"]
        best_sim = best["similarity"]

        if len(ranked) > 1:
            second_best = ranked[1]
            second_id = second_best["student_id"]
            second_sim = second_best["similarity"]
            margin = best_sim - second_sim
        else:
            second_id = None
            second_sim = 0.0
            margin = best_sim

        # 5. Dual-Condition Recognition Decision:
        # Condition 1: Absolute similarity meets or exceeds threshold
        # Condition 2: Margin over second-best student meets margin threshold (preventing identity confusion)
        passes_similarity = (best_sim >= self.threshold)
        passes_margin = (margin >= self.margin_threshold) if len(ranked) > 1 else True

        if passes_similarity and passes_margin:
            status = RecognitionStatus.MATCH
            matched_id = best_id
            matched_name = best_name
        else:
            status = RecognitionStatus.UNKNOWN
            matched_id = None
            matched_name = None

        return RecognitionResult(
            face_id=face_id,
            matched_student_id=matched_id,
            matched_student_name=matched_name,
            similarity=round(float(best_sim), 4),
            status=status,
            threshold=self.threshold,
            embedding_model=self.model_name,
            processing_time_ms=proc_time_ms,
            bbox=bbox,
            quality_metrics=qm,
            second_best_student_id=second_id,
            second_best_similarity=round(float(second_sim), 4),
            margin=round(float(margin), 4)
        )

    def recognize_batch(self, ready_faces: List[RecognitionReadyFace]) -> List[RecognitionResult]:
        """
        Recognizes multiple detected faces independently.
        One failed embedding/face will NEVER crash or halt the recognition of remaining faces.
        """
        results: List[RecognitionResult] = []
        if not ready_faces:
            return results

        for face in ready_faces:
            try:
                res = self.recognize_face(face)
                results.append(res)
            except Exception as e:
                logger.error(f"Unexpected error recognizing face: {e}")
                results.append(RecognitionResult(
                    face_id=getattr(face.quality_metrics, 'face_id', 'err_face'),
                    matched_student_id=None,
                    matched_student_name=None,
                    similarity=0.0,
                    status=RecognitionStatus.ERROR,
                    threshold=self.threshold,
                    embedding_model=self.model_name,
                    processing_time_ms=0.0,
                    bbox=getattr(face, 'original_bbox', []),
                    quality_metrics=getattr(face, 'quality_metrics', None)
                ))

        return results
