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
        threshold: Optional[float] = None
    ):
        self.settings = get_settings().recognition
        self.embedder = embedder or ArcFaceEmbedder()
        self.vector_store = vector_store or FaissVectorStore()
        self.threshold = threshold if threshold is not None else self.settings.similarity_threshold
        self.model_name = self.settings.model_name

    def set_threshold(self, new_threshold: float):
        """Allows dynamically updating the recognition threshold for calibration/tuning."""
        if not (0.0 <= new_threshold <= 1.0):
            raise ValueError(f"Similarity threshold must be between 0.0 and 1.0, got {new_threshold}.")
        self.threshold = float(new_threshold)
        logger.info(f"FaceRecognizer threshold updated to {self.threshold:.3f}")

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
                quality_metrics=qm
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
                quality_metrics=qm
            )

        # 2. Query FAISS Vector Store
        try:
            matches = self.vector_store.search(embedding, top_k=1)
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
                quality_metrics=qm
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
                quality_metrics=qm
            )

        # 4. Unknown Rejection & Threshold Decision
        top_sim, top_meta = matches[0]

        if top_sim >= self.threshold:
            status = RecognitionStatus.MATCH
            matched_id = top_meta.get("student_id")
            matched_name = top_meta.get("student_name")
        else:
            status = RecognitionStatus.UNKNOWN
            matched_id = None
            matched_name = None

        return RecognitionResult(
            face_id=face_id,
            matched_student_id=matched_id,
            matched_student_name=matched_name,
            similarity=float(top_sim),
            status=status,
            threshold=self.threshold,
            embedding_model=self.model_name,
            processing_time_ms=proc_time_ms,
            bbox=bbox,
            quality_metrics=qm
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
