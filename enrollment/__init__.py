"""Phase 10 — Real Student Enrollment & Identity Database Package."""

from .data_importer import StudentDataImporter, StudentImportRecord, ImporterValidationError
from .validation_pipeline import EnrollmentValidationPipeline, FaceValidationResult, EnrollmentStatus
from .enrollment_service import EnrollmentService, EnrollmentReport

# Backwards/alternate aliases
EnrollmentFaceValidator = EnrollmentValidationPipeline
EnrollmentFaceResult = FaceValidationResult

__all__ = [
    "StudentDataImporter",
    "StudentImportRecord",
    "ImporterValidationError",
    "EnrollmentValidationPipeline",
    "FaceValidationResult",
    "EnrollmentStatus",
    "EnrollmentFaceValidator",
    "EnrollmentFaceResult",
    "EnrollmentService",
    "EnrollmentReport",
]
