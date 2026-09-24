"""
Phase 10 Comprehensive Test Suite — Real Student Enrollment & Identity Database.

Covers all 30 mandatory requirements:
 1. Valid student record
 2. Missing Register Number
 3. Missing Name
 4. Missing Class
 5. Missing Photo
 6. Invalid photo path
 7. Corrupt image
 8. No face
 9. Multiple faces
10. Low-quality face
11. Valid face
12. Embedding generation
13. Embedding dimension validation
14. NaN embedding
15. Inf embedding
16. Embedding normalization
17. SQLite insertion
18. FAISS insertion
19. FAISS-to-student mapping
20. Duplicate Register Number
21. Duplicate enrollment
22. Student update
23. Student removal
24. Student re-enrollment
25. FAISS rebuild
26. SQLite/FAISS consistency
27. Unknown rejection
28. Known student recognition
29. Multiple students
30. Enrollment failure recovery
Plus:
31. Batch enrollment on Real Student Dataset (DOCX)
32. REST API endpoints for enrollment
"""

import os
import sys
import io
import time
import tempfile
import zipfile
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from core.vector_store import FaissVectorStore
from core.face_embedder import ArcFaceEmbedder
from core.detector import YOLOv8FaceDetector
from core.face_quality import FaceQualityAssessor
from core.face_alignment import FaceAligner
from core.recognizer import FaceRecognizer
from core.schemas import RecognitionReadyFace, FaceQualityResult, RecognitionStatus

from enrollment.data_importer import StudentDataImporter, StudentImportRecord, ImporterValidationError
from enrollment.validation_pipeline import EnrollmentValidationPipeline, FaceValidationResult, EnrollmentStatus
from enrollment.enrollment_service import EnrollmentService, EnrollmentReport
from app.main import app
from app.state import get_app_state


def create_synthetic_face_image(num_faces: int = 1, face_size: int = 80, blur: bool = False) -> np.ndarray:
    """
    Creates an image with synthetic face(s) that can be detected or simulated.
    """
    img = np.ones((300, 300, 3), dtype=np.uint8) * 180  # Grey background
    for i in range(num_faces):
        cx = 100 + i * 110
        cy = 150
        if cx + face_size // 2 >= 300:
            continue
        # Draw face oval
        cv2.ellipse(img, (cx, cy), (face_size // 2, face_size // 2 + 15), 0, 0, 360, (220, 200, 180), -1)
        # Eyes
        cv2.circle(img, (cx - 15, cy - 10), 5, (50, 50, 50), -1)
        cv2.circle(img, (cx + 15, cy - 10), 5, (50, 50, 50), -1)
        # Nose
        cv2.line(img, (cx, cy - 5), (cx, cy + 10), (100, 100, 100), 2)
        # Mouth
        cv2.ellipse(img, (cx, cy + 22), (15, 6), 0, 0, 180, (50, 50, 150), 2)
    if blur:
        img = cv2.GaussianBlur(img, (25, 25), 10)
    return img


def encode_image(img: np.ndarray, ext: str = ".jpg") -> bytes:
    _, buf = cv2.imencode(ext, img)
    return buf.tobytes()


def create_normalized_face_tensor(seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.uniform(-1.0, 1.0, size=(1, 3, 112, 112)).astype(np.float32)


@pytest.fixture
def temp_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_enrollment.sqlite")
        idx_path = os.path.join(tmpdir, "test_enrollment.bin")
        db = DatabaseManager(db_path=db_path)
        store = FaissVectorStore(embedding_dim=512, index_path=idx_path)
        embedder = ArcFaceEmbedder()
        service = EnrollmentService(
            db=db,
            vector_store=store,
            embedder=embedder,
            conflict_threshold=0.95
        )
        yield db, store, embedder, service, tmpdir


# ---------------------------------------------------------------------------
# 1. Valid Student Record (Data Importer)
# ---------------------------------------------------------------------------
def test_01_valid_student_record(temp_env):
    db, store, embedder, service, tmpdir = temp_env
    importer = StudentDataImporter()

    csv_content = (
        "Reg No,Name,Class,Photo\n"
        "922524243001,John Doe,III CSE-A,photo1.jpg\n"
    )
    csv_path = os.path.join(tmpdir, "valid.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_content)

    records = importer.import_csv(csv_path)
    assert len(records) == 1
    rec = records[0]
    assert rec.register_no == "922524243001"
    assert rec.name == "John Doe"
    assert rec.class_section == "III CSE-A"
    assert rec.photo_filenames == ["photo1.jpg"]


# ---------------------------------------------------------------------------
# 2. Missing Register Number
# ---------------------------------------------------------------------------
def test_02_missing_register_number(temp_env):
    _, _, _, _, tmpdir = temp_env
    importer = StudentDataImporter()

    # Missing Reg No column
    csv_content = "Student Name,Class\nJane Doe,III CSE-A\n"
    csv_path = os.path.join(tmpdir, "no_regno_col.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_content)

    with pytest.raises(ImporterValidationError) as exc:
        importer.import_csv(csv_path)
    assert "register_no" in str(exc.value)

    # Empty value in Reg No
    csv_content_empty = "Reg No,Name,Class\n,Jane Doe,III CSE-A\n"
    csv_path_empty = os.path.join(tmpdir, "empty_regno.csv")
    with open(csv_path_empty, "w", encoding="utf-8") as f:
        f.write(csv_content_empty)

    with pytest.raises(ImporterValidationError) as exc2:
        importer.import_csv(csv_path_empty)
    assert "Missing required 'register_no'" in str(exc2.value)


# ---------------------------------------------------------------------------
# 3. Missing Name
# ---------------------------------------------------------------------------
def test_03_missing_name(temp_env):
    _, _, _, _, tmpdir = temp_env
    importer = StudentDataImporter()

    csv_content = "Reg No,Name,Class\n922524243002,,III CSE-A\n"
    csv_path = os.path.join(tmpdir, "no_name.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_content)

    with pytest.raises(ImporterValidationError) as exc:
        importer.import_csv(csv_path)
    assert "Missing required 'name'" in str(exc.value)


# ---------------------------------------------------------------------------
# 4. Missing Class (Handled gracefully with fallback / default)
# ---------------------------------------------------------------------------
def test_04_missing_class(temp_env):
    _, _, _, _, tmpdir = temp_env
    importer = StudentDataImporter()

    # Header has no class column: should default to 'Unknown' without crash
    csv_content = "Reg No,Name\n922524243003,Bob Smith\n"
    csv_path = os.path.join(tmpdir, "no_class.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_content)

    records = importer.import_csv(csv_path)
    assert len(records) == 1
    assert records[0].class_section == "Unknown"


# ---------------------------------------------------------------------------
# 5. Missing Photo
# ---------------------------------------------------------------------------
def test_05_missing_photo(temp_env):
    db, store, embedder, service, _ = temp_env

    # Record with 0 photos
    rec = StudentImportRecord(
        register_no="922524243004",
        name="No Photo Student",
        class_section="III AI&D",
        photos=[]
    )
    report = service.enroll_batch([rec])
    assert report.total_records == 1
    assert report.successfully_enrolled == 0
    assert report.counts.get(EnrollmentStatus.NO_PHOTO) == 1
    assert len(report.failures) == 1
    assert report.failures[0]["status"] == EnrollmentStatus.NO_PHOTO


# ---------------------------------------------------------------------------
# 6. Invalid Photo Path
# ---------------------------------------------------------------------------
def test_06_invalid_photo_path(temp_env):
    _, _, _, _, tmpdir = temp_env
    importer = StudentDataImporter()

    csv_content = "Reg No,Name,Class,Photo\n922524243005,Invalid Path Student,III AI&D,non_existent_file.jpg\n"
    csv_path = os.path.join(tmpdir, "invalid_path.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_content)

    # Importer parses the reference safely; photos bytes will be empty because file not on disk
    records = importer.import_csv(csv_path)
    assert len(records) == 1
    assert len(records[0].photos) == 0  # not resolved to bytes


# ---------------------------------------------------------------------------
# 7. Corrupt Image
# ---------------------------------------------------------------------------
def test_07_corrupt_image(temp_env):
    db, store, embedder, service, _ = temp_env
    corrupt_bytes = b"NOT_A_REAL_IMAGE_CORRUPTED_HEADER_DATA"

    rec = StudentImportRecord(
        register_no="922524243006",
        name="Corrupt Image Student",
        class_section="III AI&D",
        photos=[corrupt_bytes]
    )
    report = service.enroll_batch([rec])
    assert report.counts.get(EnrollmentStatus.CORRUPT_IMAGE) == 1
    assert report.successfully_enrolled == 0


# ---------------------------------------------------------------------------
# 8. No Face Detected
# ---------------------------------------------------------------------------
def test_08_no_face_detected():
    pipeline = EnrollmentValidationPipeline()
    # Blank black image has 0 faces
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
    res = pipeline.validate_image(encode_image(blank_img))

    assert res.is_valid is False
    assert res.status == EnrollmentStatus.NO_FACE
    assert res.num_faces_detected == 0


# ---------------------------------------------------------------------------
# 9. Multiple Faces Detected
# ---------------------------------------------------------------------------
def test_09_multiple_faces_detected():
    pipeline = EnrollmentValidationPipeline()
    # Mock detector to simulate multiple faces returned
    class MultiFaceMockDetector:
        def detect_faces(self, img, **kwargs):
            return [
                {"bbox": [10, 10, 50, 50], "confidence": 0.95, "landmarks": np.zeros((5, 2))},
                {"bbox": [60, 60, 100, 100], "confidence": 0.92, "landmarks": np.zeros((5, 2))}
            ]
    mock_pipeline = EnrollmentValidationPipeline(detector=MultiFaceMockDetector())
    test_img = np.zeros((200, 200, 3), dtype=np.uint8)
    res = mock_pipeline.validate_image(encode_image(test_img))

    assert res.is_valid is False
    assert res.status == EnrollmentStatus.MULTIPLE_FACES
    assert res.num_faces_detected == 2


# ---------------------------------------------------------------------------
# 10. Low-Quality Face Gating
# ---------------------------------------------------------------------------
def test_10_low_quality_face_gating():
    class LowQualityMockAssessor:
        def assess(self, det, frame, face_id):
            return FaceQualityResult(
                face_id=face_id,
                bbox=[0, 0, 10, 10],
                width=10, height=10, area=100,
                sharpness=10.0, brightness=20.0,
                quality_status="REJECTED",
                rejection_reason="Face area 100 px < 1600 px min"
            )
    class SingleFaceMockDetector:
        def detect_faces(self, img, **kwargs):
            return [{"bbox": [0, 0, 10, 10], "confidence": 0.99, "landmarks": np.zeros((5, 2))}]

    pipeline = EnrollmentValidationPipeline(
        detector=SingleFaceMockDetector(),
        assessor=LowQualityMockAssessor()
    )
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = pipeline.validate_image(encode_image(test_img))

    assert res.is_valid is False
    assert res.status == EnrollmentStatus.LOW_QUALITY
    assert "Face area" in res.rejection_reason


# ---------------------------------------------------------------------------
# 11. Valid Face Pipeline Execution
# ---------------------------------------------------------------------------
def test_11_valid_face_pipeline():
    class ValidMockAssessor:
        def assess(self, det, frame, face_id):
            return FaceQualityResult(
                face_id=face_id,
                bbox=[20, 20, 120, 120],
                width=100, height=100, area=10000,
                sharpness=180.0, brightness=120.0,
                quality_status="RECOGNITION_READY",
                keypoints=[[35, 45], [75, 45], [55, 65], [40, 85], [70, 85]]
            )
    class SingleFaceMockDetector:
        def detect_faces(self, img, **kwargs):
            return [{
                "bbox": [20, 20, 120, 120],
                "confidence": 0.99,
                "landmarks": np.array([[35, 45], [75, 45], [55, 65], [40, 85], [70, 85]])
            }]

    pipeline = EnrollmentValidationPipeline(
        detector=SingleFaceMockDetector(),
        assessor=ValidMockAssessor()
    )
    test_img = np.ones((200, 200, 3), dtype=np.uint8) * 150
    res = pipeline.validate_image(encode_image(test_img))

    assert res.is_valid is True
    assert res.status == EnrollmentStatus.PASS
    assert res.aligned_tensor is not None
    assert res.aligned_tensor.shape == (1, 3, 112, 112)
    assert res.aligned_tensor.dtype == np.float32


# ---------------------------------------------------------------------------
# 12. Embedding Generation
# ---------------------------------------------------------------------------
def test_12_embedding_generation(temp_env):
    _, _, embedder, _, _ = temp_env
    tensor = create_normalized_face_tensor(seed=12)
    emb = embedder.generate_embedding(tensor)

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (1, 512)
    assert emb.dtype == np.float32


# ---------------------------------------------------------------------------
# 13. Embedding Dimension Validation
# ---------------------------------------------------------------------------
def test_13_embedding_dimension_validation(temp_env):
    _, _, embedder, _, _ = temp_env
    assert embedder.embedding_dim == 512

    # Vector with wrong dimension
    bad_emb = np.ones((256,), dtype=np.float32)
    assert embedder.validate_embedding(bad_emb) is False


# ---------------------------------------------------------------------------
# 14. NaN Embedding Rejection
# ---------------------------------------------------------------------------
def test_14_nan_embedding_rejection(temp_env):
    _, _, embedder, _, _ = temp_env
    nan_emb = np.ones((512,), dtype=np.float32)
    nan_emb[10] = np.nan
    assert embedder.validate_embedding(nan_emb) is False

    # Also via tensor input
    nan_tensor = create_normalized_face_tensor(seed=14)
    nan_tensor[0, 0, 10, 10] = np.nan
    with pytest.raises(ValueError):
        embedder.generate_embedding(nan_tensor)


# ---------------------------------------------------------------------------
# 15. Inf Embedding Rejection
# ---------------------------------------------------------------------------
def test_15_inf_embedding_rejection(temp_env):
    _, _, embedder, _, _ = temp_env
    inf_emb = np.ones((512,), dtype=np.float32)
    inf_emb[20] = np.inf
    assert embedder.validate_embedding(inf_emb) is False

    inf_tensor = create_normalized_face_tensor(seed=15)
    inf_tensor[0, 0, 5, 5] = np.inf
    with pytest.raises(ValueError):
        embedder.generate_embedding(inf_tensor)


# ---------------------------------------------------------------------------
# 16. Embedding Normalization (Unit L2 Norm)
# ---------------------------------------------------------------------------
def test_16_embedding_normalization(temp_env):
    _, _, embedder, _, _ = temp_env
    tensor = create_normalized_face_tensor(seed=16)
    emb = embedder.generate_embedding(tensor)

    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 1e-4
    assert embedder.validate_embedding(emb) is True


# ---------------------------------------------------------------------------
# 17. SQLite Insertion
# ---------------------------------------------------------------------------
def test_17_sqlite_insertion(temp_env):
    db, _, _, _, _ = temp_env
    reg = db.add_student("STUD_17", "Alice Wonder", department="CSE", section="B", class_name="III CSE-B")
    assert reg is True

    student = db.get_student("STUD_17")
    assert student is not None
    assert student["student_name"] == "Alice Wonder"
    assert student["class_name"] == "III CSE-B"


# ---------------------------------------------------------------------------
# 18. FAISS Insertion
# ---------------------------------------------------------------------------
def test_18_faiss_insertion(temp_env):
    _, store, _, _, _ = temp_env
    vec = np.ones((512,), dtype=np.float32)
    vec /= np.linalg.norm(vec)

    meta = {"embedding_id": 1, "student_id": "STUD_18", "student_name": "Bob"}
    res = store.add_vector(vector_id=1, embedding=vec, metadata=meta)
    assert res is True
    assert store.total_vectors == 1


# ---------------------------------------------------------------------------
# 19. FAISS-to-Student Mapping
# ---------------------------------------------------------------------------
def test_19_faiss_to_student_mapping(temp_env):
    _, store, _, _, _ = temp_env
    vec = np.random.randn(512).astype(np.float32)
    vec /= np.linalg.norm(vec)

    meta = {"embedding_id": 99, "student_id": "STUD_99", "student_name": "Charlie", "reg_no": "922524243099"}
    store.add_vector(99, vec, meta)

    search_res = store.search(vec, top_k=1)
    assert len(search_res) == 1
    sim, returned_meta = search_res[0]
    assert abs(sim - 1.0) < 1e-4
    assert returned_meta["student_id"] == "STUD_99"
    assert returned_meta["student_name"] == "Charlie"


# ---------------------------------------------------------------------------
# 20. Duplicate Register Number
# ---------------------------------------------------------------------------
def test_20_duplicate_register_number(temp_env):
    _, _, _, _, tmpdir = temp_env
    importer = StudentDataImporter()

    csv_content = (
        "Reg No,Name,Class\n"
        "922524243020,Student A,III CSE-A\n"
        "922524243020,Student B,III CSE-B\n"
    )
    csv_path = os.path.join(tmpdir, "dup_regno.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_content)

    with pytest.raises(ImporterValidationError) as exc:
        importer.import_csv(csv_path)
    assert "Duplicate register_no '922524243020'" in str(exc.value)


# ---------------------------------------------------------------------------
# 21. Duplicate Enrollment Conflict Flag
# ---------------------------------------------------------------------------
def test_21_duplicate_enrollment_conflict(temp_env):
    db, store, embedder, service, _ = temp_env

    # Register Student 1
    service.register_student("STUD_21A", "Student 21A", reg_no="REG21A")
    tensor_a = create_normalized_face_tensor(seed=210)
    service.enroll_face_sample("STUD_21A", tensor_a)

    # Register Student 2
    service.register_student("STUD_21B", "Student 21B", reg_no="REG21B")
    # Try to enroll identical tensor for Student 2
    success, emb_id, msg = service.enroll_face_sample("STUD_21B", tensor_a)
    assert success is False
    assert "Identity conflict" in msg


# ---------------------------------------------------------------------------
# 22. Student Update
# ---------------------------------------------------------------------------
def test_22_student_update(temp_env):
    db, store, embedder, service, _ = temp_env
    service.register_student("STUD_22", "Original Name", reg_no="REG22", class_section="Class 1")

    ok = service.update_student("STUD_22", student_name="Updated Name", class_section="Class 2")
    assert ok is True

    student = db.get_student("STUD_22")
    assert student["student_name"] == "Updated Name"
    assert student["class_name"] == "Class 2"


# ---------------------------------------------------------------------------
# 23. Student Removal
# ---------------------------------------------------------------------------
def test_23_student_removal(temp_env):
    db, store, embedder, service, _ = temp_env
    service.register_student("STUD_23", "To Be Removed", reg_no="REG23")
    tensor = create_normalized_face_tensor(seed=23)
    service.enroll_face_sample("STUD_23", tensor)
    assert store.total_vectors == 1

    ok = service.remove_student("STUD_23")
    assert ok is True

    # Check student is gone from SQLite and FAISS
    assert db.get_student("STUD_23") is None
    assert store.total_vectors == 0


# ---------------------------------------------------------------------------
# 24. Student Re-enrollment
# ---------------------------------------------------------------------------
def test_24_student_re_enrollment(temp_env):
    db, store, embedder, service, _ = temp_env
    service.register_student("STUD_24", "ReEnroll Student", reg_no="REG24")

    # Photo 1
    tensor1 = create_normalized_face_tensor(seed=241)
    service.enroll_face_sample("STUD_24", tensor1, sample_label="photo_old")
    assert store.total_vectors == 1

    # Re-enroll with new Photo 2
    tensor2 = create_normalized_face_tensor(seed=242)
    success, count, msg = service.re_enroll_student("STUD_24", [tensor2], labels=["photo_new"])
    assert success is True
    assert count == 1
    assert store.total_vectors == 1

    # Search with tensor2 should match STUD_24
    emb2 = embedder.generate_embedding(tensor2)
    matches = store.search(emb2, top_k=1)
    assert matches[0][1]["student_id"] == "STUD_24"
    assert matches[0][1]["sample_label"] == "photo_new"


# ---------------------------------------------------------------------------
# 25. FAISS Rebuild from SQLite
# ---------------------------------------------------------------------------
def test_25_faiss_rebuild(temp_env):
    db, store, embedder, service, _ = temp_env
    service.register_student("STUD_25A", "Student 25A", reg_no="REG25A")
    service.register_student("STUD_25B", "Student 25B", reg_no="REG25B")

    tensor_a = create_normalized_face_tensor(seed=251)
    tensor_b = create_normalized_face_tensor(seed=252)

    service.enroll_face_sample("STUD_25A", tensor_a)
    service.enroll_face_sample("STUD_25B", tensor_b)
    assert store.total_vectors == 2

    # Clear FAISS index completely in-memory
    store.clear()
    assert store.total_vectors == 0

    # Rebuild from SQLite
    rebuilt_count = service.rebuild_faiss_index()
    assert rebuilt_count == 2
    assert store.total_vectors == 2


# ---------------------------------------------------------------------------
# 26. SQLite/FAISS Consistency Check
# ---------------------------------------------------------------------------
def test_26_sqlite_faiss_consistency(temp_env):
    db, store, embedder, service, _ = temp_env
    service.register_student("STUD_26", "Student 26", reg_no="REG26")
    tensor = create_normalized_face_tensor(seed=26)
    service.enroll_face_sample("STUD_26", tensor)

    # Clean state
    status = service.verify_index_consistency()
    assert status["consistent"] is True
    assert status["db_embeddings_count"] == 1
    assert status["faiss_vectors_count"] == 1

    # Inject discrepancy by resetting store
    store.clear()
    status_discrepant = service.verify_index_consistency()
    assert status_discrepant["consistent"] is False


# ---------------------------------------------------------------------------
# 27. Unknown Rejection
# ---------------------------------------------------------------------------
def test_27_unknown_rejection(temp_env):
    db, store, embedder, service, _ = temp_env
    recognizer = FaceRecognizer(embedder=embedder, vector_store=store, threshold=0.85)

    # Enroll Known
    service.register_student("KNOWN_STUD", "Known Student", reg_no="REG_KNOWN")
    tensor_known = create_normalized_face_tensor(seed=271)
    service.enroll_face_sample("KNOWN_STUD", tensor_known)

    # Query with Unknown
    tensor_unknown = create_normalized_face_tensor(seed=272)
    face_query = RecognitionReadyFace(
        quality_metrics=FaceQualityResult(
            face_id="q_unk", bbox=[0, 0, 10, 10], width=10, height=10, area=100,
            sharpness=100, brightness=100, quality_status="RECOGNITION_READY"
        ),
        original_bbox=[0, 0, 10, 10],
        aligned_face_tensor=tensor_unknown,
        timestamp=time.time()
    )

    result = recognizer.recognize_face(face_query)
    assert result.status == RecognitionStatus.UNKNOWN
    assert result.matched_student_id is None


# ---------------------------------------------------------------------------
# 28. Known Student Recognition
# ---------------------------------------------------------------------------
def test_28_known_student_recognition(temp_env):
    db, store, embedder, service, _ = temp_env
    recognizer = FaceRecognizer(embedder=embedder, vector_store=store, threshold=0.65)

    service.register_student("KNOWN_28", "John Doe", reg_no="REG_28")
    tensor_known = create_normalized_face_tensor(seed=28)
    service.enroll_face_sample("KNOWN_28", tensor_known)

    face_query = RecognitionReadyFace(
        quality_metrics=FaceQualityResult(
            face_id="q_known", bbox=[0, 0, 10, 10], width=10, height=10, area=100,
            sharpness=100, brightness=100, quality_status="RECOGNITION_READY"
        ),
        original_bbox=[0, 0, 10, 10],
        aligned_face_tensor=tensor_known,
        timestamp=time.time()
    )

    result = recognizer.recognize_face(face_query)
    assert result.status == RecognitionStatus.MATCH
    assert result.matched_student_id == "KNOWN_28"
    assert result.matched_student_name == "John Doe"
    assert result.similarity >= 0.99


# ---------------------------------------------------------------------------
# 29. Multiple Students Enrollment & Separation
# ---------------------------------------------------------------------------
def test_29_multiple_students(temp_env):
    db, store, embedder, service, _ = temp_env
    recognizer = FaceRecognizer(embedder=embedder, vector_store=store, threshold=0.85)

    students = [("S_1", "Student One", 101), ("S_2", "Student Two", 102), ("S_3", "Student Three", 103)]
    tensors = {}
    for s_id, s_name, seed in students:
        service.register_student(s_id, s_name)
        t = create_normalized_face_tensor(seed=seed)
        tensors[s_id] = t
        service.enroll_face_sample(s_id, t)

    assert store.total_vectors == 3

    # Query each student
    for s_id, s_name, _ in students:
        q = RecognitionReadyFace(
            quality_metrics=FaceQualityResult(face_id=f"q_{s_id}", bbox=[0,0,10,10], width=10, height=10, area=100, sharpness=100, brightness=100, quality_status="RECOGNITION_READY"),
            original_bbox=[0,0,10,10],
            aligned_face_tensor=tensors[s_id],
            timestamp=time.time()
        )
        res = recognizer.recognize_face(q)
        assert res.status == RecognitionStatus.MATCH
        assert res.matched_student_id == s_id


# ---------------------------------------------------------------------------
# 30. Enrollment Failure Recovery
# ---------------------------------------------------------------------------
def test_30_enrollment_failure_recovery(temp_env):
    db, store, embedder, service, _ = temp_env

    # 1. Enrolling for unregistered student gracefully fails without corrupting database or FAISS
    t = create_normalized_face_tensor(seed=30)
    success, _, msg = service.enroll_face_sample("NON_EXISTENT", t)
    assert success is False
    assert "not registered" in msg
    assert store.total_vectors == 0

    # 2. Enrolling invalid tensor with NaN does not corrupt vector store
    service.register_student("STUD_30", "Student 30")
    t_nan = np.full((1, 3, 112, 112), np.nan, dtype=np.float32)
    success_nan, _, msg_nan = service.enroll_face_sample("STUD_30", t_nan)
    assert success_nan is False
    assert store.total_vectors == 0

    # 3. Clean recovery and successful subsequent enrollment
    success, emb_id, _ = service.enroll_face_sample("STUD_30", t)
    assert success is True
    assert store.total_vectors == 1


# ---------------------------------------------------------------------------
# 31. Batch Import on Real DOCX Dataset (If file exists)
# ---------------------------------------------------------------------------
def test_31_real_docx_import_validation():
    docx_path = r"C:\Users\kanis\Downloads\Student Sample.docx"
    if not os.path.exists(docx_path):
        pytest.skip("Real student docx sample file not present at path.")

    importer = StudentDataImporter()
    records = importer.import_docx(docx_path)
    assert len(records) > 0

    # Verify fields mapped properly
    first = records[0]
    assert first.register_no != ""
    assert first.name != ""
    assert len(first.photos) > 0  # photo stream extracted from DOCX drawingML


# ---------------------------------------------------------------------------
# 32. Enrollment REST APIs
# ---------------------------------------------------------------------------
def test_32_enrollment_api_endpoints(temp_env):
    db, store, embedder, service, tmpdir = temp_env

    # Point global app state to our test service
    state = get_app_state()
    state._enrollment_service = service
    state.db = db
    state._vector_store = store

    client = TestClient(app)
    headers = {"X-Operator-Token": "smartclass_operator_2026"}

    # 1. GET /api/enrollment/status
    res = client.get("/api/enrollment/status")
    assert res.status_code == 200
    res_json = res.json()
    assert res_json["success"] is True
    data = res_json["data"]
    assert data["consistent"] is True

    # 2. POST /api/enrollment/student (Manual single student register)
    payload = {
        "student_id": "API_STUD_01",
        "student_name": "API Student",
        "register_no": "REG_API_01",
        "class_section": "III AI&D B",
        "department": "AI&DS",
        "section": "B"
    }
    res_reg = client.post("/api/enrollment/student", json=payload, headers=headers)
    assert res_reg.status_code == 200
    assert res_reg.json()["success"] is True

    # 3. POST /api/enrollment/rebuild
    res_rebuild = client.post("/api/enrollment/rebuild", headers=headers)
    assert res_rebuild.status_code == 200
    assert res_rebuild.json()["success"] is True

    # 4. DELETE /api/enrollment/student/API_STUD_01
    res_del = client.delete("/api/enrollment/student/API_STUD_01", headers=headers)
    assert res_del.status_code == 200
    assert res_del.json()["success"] is True


# ---------------------------------------------------------------------------
# 33. XLSX Document Import Validation (openpyxl)
# ---------------------------------------------------------------------------
def test_33_xlsx_import_validation(temp_env):
    _, _, _, _, tmpdir = temp_env
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Students"
    ws.append(["Reg No", "Name", "Class", "Photo Reference"])
    ws.append(["922524243999", "Excel Student", "III AI&DS-B", "student.jpg"])

    xlsx_path = os.path.join(tmpdir, "test_students.xlsx")
    wb.save(xlsx_path)

    importer = StudentDataImporter()
    records = importer.import_xlsx(xlsx_path)

    assert len(records) == 1
    assert records[0].register_no == "922524243999"
    assert records[0].name == "Excel Student"
    assert records[0].class_section == "III AI&DS-B"
    assert records[0].department == "AI&DS"
    assert records[0].section == "B"


# ---------------------------------------------------------------------------
# 34. Camera Multi-Shot Enrollment (Phase 10.10)
# ---------------------------------------------------------------------------
def test_34_camera_multi_shot_enrollment(temp_env):
    db, store, embedder, service, _ = temp_env

    # Simulate 3 camera captures of the same student
    frame1 = create_synthetic_face_image(num_faces=1, face_size=80)
    frame2 = create_synthetic_face_image(num_faces=1, face_size=85)
    frame3 = create_synthetic_face_image(num_faces=1, face_size=75)

    # Mock validation pipeline to simulate high-quality face captures
    fake_tensor1 = create_normalized_face_tensor(seed=101)
    fake_tensor2 = create_normalized_face_tensor(seed=102)
    fake_tensor3 = create_normalized_face_tensor(seed=103)

    mock_res1 = FaceValidationResult(is_valid=True, status=EnrollmentStatus.PASS, quality_score=0.95, aligned_tensor=fake_tensor1)
    mock_res2 = FaceValidationResult(is_valid=True, status=EnrollmentStatus.PASS, quality_score=0.92, aligned_tensor=fake_tensor2)
    mock_res3 = FaceValidationResult(is_valid=True, status=EnrollmentStatus.PASS, quality_score=0.96, aligned_tensor=fake_tensor3)

    service.validation_pipeline.validate_photo = lambda image_input, student_id="temp", sample_label="frontal": (
        mock_res1 if "1" in sample_label else (mock_res2 if "2" in sample_label else mock_res3)
    )

    success, emb_ids, errors = service.enroll_from_camera_samples(
        student_id="922524243099",
        student_name="Camera Student",
        frames=[frame1, frame2, frame3],
        labels=["shot_1", "shot_2", "shot_3"],
        class_section="III AI&DS-B"
    )

    assert success is True
    assert len(emb_ids) == 3
    assert len(errors) == 0

    # Verify student has 3 distinct embeddings mapped to their ID
    student_embs = db.get_embeddings_for_student("922524243099")
    assert len(student_embs) == 3
    assert store.total_vectors == 3


# ---------------------------------------------------------------------------
# 35. Unknown Protection: Never Force Nearest Below Threshold (Phase 10.9)
# ---------------------------------------------------------------------------
def test_35_unknown_protection_below_threshold(temp_env):
    db, store, embedder, service, _ = temp_env

    # Enroll one known student
    tensor = create_normalized_face_tensor(seed=55)
    service.register_student("KNOWN_STUDENT", "Known Student")
    success, emb_id, _ = service.enroll_face_sample("KNOWN_STUDENT", tensor)
    assert success is True
    store.save_index()

    # Search with an orthogonal/dissimilar vector
    # Cosine similarity between nearly orthogonal 512-dim vectors is close to 0.0
    rng = np.random.RandomState(9999)
    dissimilar_emb = rng.uniform(-1.0, 1.0, size=(512,)).astype(np.float32)
    dissimilar_emb /= np.linalg.norm(dissimilar_emb)

    matches = store.search(dissimilar_emb, top_k=1)
    assert len(matches) > 0
    top_similarity, meta = matches[0]

    # Verify similarity is well below recognition threshold (0.65)
    assert top_similarity < 0.65

    # With FaceRecognizer, this must produce UNKNOWN, never force match
    from core.recognizer import FaceRecognizer
    from core.schemas import RecognitionReadyFace
    recognizer = FaceRecognizer(vector_store=store, threshold=0.65)
    face_req = RecognitionReadyFace(
        quality_metrics=FaceQualityResult(
            face_id="q_unk", bbox=[0, 0, 10, 10], width=10, height=10, area=100,
            sharpness=100, brightness=100, quality_status="RECOGNITION_READY"
        ),
        original_bbox=[0, 0, 10, 10],
        aligned_face_tensor=tensor,
        timestamp=time.time()
    )
    # Mock embedder to return dissimilar_emb
    recognizer.embedder.generate_embedding = lambda t: dissimilar_emb

    result = recognizer.recognize_face(face_req)
    assert result.status == RecognitionStatus.UNKNOWN
    assert result.matched_student_id is None

