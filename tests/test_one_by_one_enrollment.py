"""
Phase 10: Focused Test Suite for New One-By-One Student Face Enrollment.
Completely independent from previous student documents/databases.
Tests:
 1. New student creation
 2. One-student enrollment
 3. Multiple face rejection
 4. No-face rejection
 5. Low-quality face rejection
 6. Embedding generation
 7. FAISS insertion
 8. Student-to-embedding mapping
 9. Duplicate register number rejection
10. Enrollment completion & atomic rollback
11. Final Acceptance Test: TEST001 Test Student retrieval via recognition pipeline
"""

import os
import shutil
import tempfile
import cv2
import numpy as np
import pytest

from database.new_enrollment_db import NewEnrollmentDatabase
from enrollment.one_by_one_service import OneByOneEnrollmentService, FaceSampleValidationResult
from core.vector_store import FaissVectorStore
from core.face_embedder import ArcFaceEmbedder
from core.face_alignment import FaceAligner
from core.face_quality import FaceQualityAssessor
from core.recognizer import FaceRecognizer
from core.schemas import RecognitionReadyFace, RecognitionStatus, FaceQualityResult


def make_synthetic_face(w: int = 160, h: int = 200, brightness: int = 150, blur: bool = False) -> np.ndarray:
    """Creates a synthetic face image with discernible facial features."""
    img = np.full((h, w, 3), brightness, dtype=np.uint8)
    feat_color = (20, 20, 20)
    # Eyes
    cv2.circle(img, (int(w * 0.35), int(h * 0.35)), int(w * 0.08), feat_color, -1)
    cv2.circle(img, (int(w * 0.65), int(h * 0.35)), int(w * 0.08), feat_color, -1)
    # Nose
    cv2.line(img, (int(w * 0.5), int(h * 0.45)), (int(w * 0.5), int(h * 0.6)), feat_color, 2)
    # Mouth
    cv2.rectangle(img, (int(w * 0.35), int(h * 0.72)), (int(w * 0.65), int(h * 0.8)), feat_color, -1)

    if blur:
        img = cv2.GaussianBlur(img, (25, 25), 0)
    return img


class MockDetector:
    """Mock detector to reliably test single-face, multi-face, and no-face branches."""
    def __init__(self, mode="single"):
        self.mode = mode

    def detect_faces(self, frame):
        h, w = frame.shape[:2]
        if self.mode == "none":
            return []
        elif self.mode == "multiple":
            return [
                {
                    "bbox": [50, 50, 150, 180],
                    "face_area": 13000,
                    "keypoints": [[80, 90], [120, 90], [100, 120], [85, 150], [115, 150]]
                },
                {
                    "bbox": [200, 50, 300, 180],
                    "face_area": 13000,
                    "keypoints": [[230, 90], [270, 90], [250, 120], [235, 150], [265, 150]]
                }
            ]
        else: # single
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            return [{
                "bbox": [margin_x, margin_y, w - margin_x, h - margin_y],
                "face_area": (w - 2 * margin_x) * (h - 2 * margin_y),
                "keypoints": [
                    [int(w * 0.35), int(h * 0.35)],
                    [int(w * 0.65), int(h * 0.35)],
                    [int(w * 0.50), int(h * 0.50)],
                    [int(w * 0.38), int(h * 0.75)],
                    [int(w * 0.62), int(h * 0.75)]
                ]
            }]


@pytest.fixture
def temp_env():
    """Isolated temporary test directory for database, FAISS, and photos."""
    tmp_dir = tempfile.mkdtemp(prefix="phase10_test_")
    db_path = os.path.join(tmp_dir, "test_enrollment.sqlite")
    faiss_path = os.path.join(tmp_dir, "test_faiss.bin")
    storage_dir = os.path.join(tmp_dir, "photos")

    db = NewEnrollmentDatabase(db_path=db_path)
    store = FaissVectorStore(embedding_dim=512, index_path=faiss_path)
    embedder = ArcFaceEmbedder()

    yield {
        "tmp_dir": tmp_dir,
        "db": db,
        "store": store,
        "embedder": embedder,
        "storage_dir": storage_dir
    }

    shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 1: New student creation
# ---------------------------------------------------------------------------
def test_1_new_student_creation(temp_env):
    db: NewEnrollmentDatabase = temp_env["db"]
    dummy_vec = np.random.randn(512).astype(np.float32)
    dummy_vec = dummy_vec / np.linalg.norm(dummy_vec)

    emb_ids = db.save_enrollment_atomic(
        student_id="STU_922524243069",
        register_number="922524243069",
        name="KANISH M",
        class_name="III",
        department="AI&D",
        section="S-B",
        embeddings=[dummy_vec]
    )

    assert len(emb_ids) == 1
    assert db.get_student_count() == 1

    student = db.get_student_by_register_number("922524243069")
    assert student is not None
    assert student["student_id"] == "STU_922524243069"
    assert student["register_number"] == "922524243069"
    assert student["name"] == "KANISH M"
    assert student["class"] == "III"
    assert student["department"] == "AI&D"
    assert student["section"] == "S-B"
    assert student["enrollment_status"] == "COMPLETED"


# ---------------------------------------------------------------------------
# Test 2: One-student enrollment
# ---------------------------------------------------------------------------
def test_2_one_student_enrollment(temp_env):
    service = OneByOneEnrollmentService(
        db=temp_env["db"],
        vector_store=temp_env["store"],
        embedder=temp_env["embedder"],
        detector=MockDetector(mode="single"),
        storage_base_dir=temp_env["storage_dir"]
    )

    # 5 good face samples
    samples = [make_synthetic_face(200, 250, brightness=130 + i * 10) for i in range(5)]

    res = service.enroll_student(
        register_number="922524243069",
        name="KANISH M",
        class_name="III",
        department="AI&D",
        section="S-B",
        sample_frames=samples
    )

    assert res["success"] is True
    assert res["register_number"] == "922524243069"
    assert res["name"] == "KANISH M"
    assert res["samples_enrolled"] == 5
    assert len(res["embedding_ids"]) == 5

    # Check files saved under data/enrollment/922524243069/
    photo_dir = res["photo_directory"]
    assert os.path.exists(photo_dir)
    assert os.path.exists(os.path.join(photo_dir, "sample_01.jpg"))
    assert os.path.exists(os.path.join(photo_dir, "sample_05.jpg"))


# ---------------------------------------------------------------------------
# Test 3: Multiple face rejection
# ---------------------------------------------------------------------------
def test_3_multiple_face_rejection(temp_env):
    service = OneByOneEnrollmentService(
        db=temp_env["db"],
        vector_store=temp_env["store"],
        embedder=temp_env["embedder"],
        detector=MockDetector(mode="multiple"),
        storage_base_dir=temp_env["storage_dir"]
    )

    frame = np.full((300, 400, 3), 150, dtype=np.uint8)
    val_res = service.validate_frame(frame)

    assert val_res.can_capture is False
    assert val_res.status == "MULTIPLE_FACES"
    assert val_res.message == "Multiple faces detected. Only one person can be enrolled at a time."
    assert val_res.num_faces == 2


# ---------------------------------------------------------------------------
# Test 4: No-face rejection
# ---------------------------------------------------------------------------
def test_4_no_face_rejection(temp_env):
    service = OneByOneEnrollmentService(
        db=temp_env["db"],
        vector_store=temp_env["store"],
        embedder=temp_env["embedder"],
        detector=MockDetector(mode="none"),
        storage_base_dir=temp_env["storage_dir"]
    )

    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    val_res = service.validate_frame(frame)

    assert val_res.can_capture is False
    assert val_res.status == "NO_FACE"
    assert val_res.message == "No face detected. Please position your face clearly."
    assert val_res.num_faces == 0


# ---------------------------------------------------------------------------
# Test 5: Low-quality face rejection
# ---------------------------------------------------------------------------
def test_5_low_quality_face_rejection(temp_env):
    service = OneByOneEnrollmentService(
        db=temp_env["db"],
        vector_store=temp_env["store"],
        embedder=temp_env["embedder"],
        detector=MockDetector(mode="single"),
        storage_base_dir=temp_env["storage_dir"]
    )

    # 1. Blurry face
    blurry_frame = make_synthetic_face(200, 250, brightness=150, blur=True)
    val_res_blur = service.validate_frame(blurry_frame)
    assert val_res_blur.can_capture is False
    assert val_res_blur.status == "LOW_QUALITY"
    assert "LOW QUALITY" in val_res_blur.quality_status
    assert "blurry" in val_res_blur.rejection_reason.lower()

    # 2. Too dark face
    dark_frame = make_synthetic_face(200, 250, brightness=10, blur=False)
    val_res_dark = service.validate_frame(dark_frame)
    assert val_res_dark.can_capture is False
    assert val_res_dark.status == "LOW_QUALITY"
    assert "dark" in val_res_dark.rejection_reason.lower()

    # 3. Too bright / overexposed face
    bright_frame = np.full((250, 200, 3), 245, dtype=np.uint8)
    val_res_bright = service.validate_frame(bright_frame)
    assert val_res_bright.can_capture is False
    assert val_res_bright.status == "LOW_QUALITY"
    assert "bright" in val_res_bright.rejection_reason.lower()



# ---------------------------------------------------------------------------
# Test 6: Embedding generation
# ---------------------------------------------------------------------------
def test_6_embedding_generation(temp_env):
    embedder: ArcFaceEmbedder = temp_env["embedder"]
    dummy_tensor = np.zeros((1, 3, 112, 112), dtype=np.float32)

    emb = embedder.generate_embedding(dummy_tensor)

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (1, 512)
    assert emb.dtype == np.float32
    assert not np.isnan(emb).any()
    assert not np.isinf(emb).any()

    # Verify unit norm
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# Test 7: FAISS insertion
# ---------------------------------------------------------------------------
def test_7_faiss_insertion(temp_env):
    store: FaissVectorStore = temp_env["store"]
    assert store.total_vectors == 0

    vec = np.random.randn(512).astype(np.float32)
    vec = vec / np.linalg.norm(vec)

    meta = {
        "student_id": "STU_TEST01",
        "register_number": "TEST01",
        "student_name": "Test Student"
    }

    success = store.add_vector(embedding_id=101, vector=vec, metadata=meta)
    assert success is True
    assert store.total_vectors == 1

    # Search
    matches = store.search(vec, top_k=1)
    assert len(matches) == 1
    sim, top_meta = matches[0]
    assert sim >= 0.99
    assert top_meta["student_id"] == "STU_TEST01"
    assert top_meta["register_number"] == "TEST01"


# ---------------------------------------------------------------------------
# Test 8: Student-to-embedding mapping
# ---------------------------------------------------------------------------
def test_8_student_to_embedding_mapping(temp_env):
    db: NewEnrollmentDatabase = temp_env["db"]
    store: FaissVectorStore = temp_env["store"]

    vec1 = np.random.randn(512).astype(np.float32)
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = np.random.randn(512).astype(np.float32)
    vec2 = vec2 / np.linalg.norm(vec2)

    emb_ids = db.save_enrollment_atomic(
        student_id="STU_922524243069",
        register_number="922524243069",
        name="KANISH M",
        class_name="III",
        department="AI&D",
        section="S-B",
        embeddings=[vec1, vec2]
    )

    for eid, v in zip(emb_ids, [vec1, vec2]):
        store.add_vector(eid, v, {
            "embedding_id": eid,
            "student_id": "STU_922524243069",
            "register_number": "922524243069",
            "student_name": "KANISH M"
        })

    # Retrieve from DB
    embeddings = db.get_embeddings_for_student("STU_922524243069")
    assert len(embeddings) == 2
    for item in embeddings:
        assert item["student_id"] == "STU_922524243069"
        assert item["register_number"] == "922524243069"
        assert item["vector"].shape == (512,)

    # FAISS search lookup resolves correctly
    matches = store.search(vec1, top_k=1)
    assert len(matches) == 1
    sim, meta = matches[0]
    assert meta["student_id"] == "STU_922524243069"
    assert meta["register_number"] == "922524243069"
    assert meta["student_name"] == "KANISH M"


# ---------------------------------------------------------------------------
# Test 9: Duplicate register number rejection
# ---------------------------------------------------------------------------
def test_9_duplicate_register_number_rejection(temp_env):
    service = OneByOneEnrollmentService(
        db=temp_env["db"],
        vector_store=temp_env["store"],
        embedder=temp_env["embedder"],
        detector=MockDetector(mode="single"),
        storage_base_dir=temp_env["storage_dir"]
    )

    samples = [make_synthetic_face(200, 250, brightness=140) for _ in range(5)]

    # 1. First enrollment succeeds
    service.enroll_student(
        register_number="922524243069",
        name="KANISH M",
        class_name="III",
        department="AI&D",
        section="S-B",
        sample_frames=samples
    )

    # 2. Check duplicate helper returns True
    is_dup, existing = service.check_duplicate_register_number("922524243069")
    assert is_dup is True
    assert existing["name"] == "KANISH M"

    # 3. Second enrollment with same register number is rejected
    with pytest.raises(ValueError) as excinfo:
        service.enroll_student(
            register_number="922524243069",
            name="ANOTHER STUDENT",
            class_name="III",
            department="AI&D",
            section="S-B",
            sample_frames=samples
        )
    assert "already enrolled" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Test 10: Enrollment completion & atomic rollback
# ---------------------------------------------------------------------------
def test_10_enrollment_completion_and_rollback(temp_env):
    service = OneByOneEnrollmentService(
        db=temp_env["db"],
        vector_store=temp_env["store"],
        embedder=temp_env["embedder"],
        detector=MockDetector(mode="single"),
        storage_base_dir=temp_env["storage_dir"]
    )

    # Only 3 samples (< 5 required)
    samples = [make_synthetic_face(200, 250, brightness=140) for _ in range(3)]

    with pytest.raises(ValueError) as excinfo:
        service.enroll_student(
            register_number="FAIL001",
            name="Failing Student",
            class_name="III",
            department="AI&D",
            section="S-B",
            sample_frames=samples
        )
    assert "At least 5 good face samples are required" in str(excinfo.value)

    # Ensure no partial student was created
    assert temp_env["db"].get_student_count() == 0
    assert temp_env["db"].get_embedding_count() == 0
    assert not os.path.exists(os.path.join(temp_env["storage_dir"], "FAIL001"))


# ---------------------------------------------------------------------------
# FINAL ACCEPTANCE TEST:
# TEST001 -> Test Student -> 5 samples -> FAISS -> New DB -> Retrieved
# ---------------------------------------------------------------------------
def test_11_final_acceptance_test(temp_env):
    """
    Perform one controlled test using a single authorized test person.
    Register No: TEST001
    Name: Test Student
    Capture 5 good samples.
    Verify: TEST001 -> Test Student -> 5 face samples -> 5 embeddings -> FAISS -> New enrollment DB.
    Then verify recognition pipeline retrieves Register No: TEST001, Name: Test Student.
    """
    db: NewEnrollmentDatabase = temp_env["db"]
    store: FaissVectorStore = temp_env["store"]
    embedder: ArcFaceEmbedder = temp_env["embedder"]

    service = OneByOneEnrollmentService(
        db=db,
        vector_store=store,
        embedder=embedder,
        detector=MockDetector(mode="single"),
        storage_base_dir=temp_env["storage_dir"]
    )

    # Generate 5 good face samples
    samples = [make_synthetic_face(200, 250, brightness=135 + i * 5) for i in range(5)]

    # Execute enrollment
    enroll_result = service.enroll_student(
        register_number="TEST001",
        name="Test Student",
        class_name="III",
        department="AI&D",
        section="S-B",
        sample_frames=samples
    )

    assert enroll_result["success"] is True
    assert enroll_result["register_number"] == "TEST001"
    assert enroll_result["name"] == "Test Student"
    assert enroll_result["samples_enrolled"] == 5

    # 1. Verify in New Enrollment Database
    db_student = db.get_student_by_register_number("TEST001")
    assert db_student is not None
    assert db_student["student_id"] == "STU_TEST01" or db_student["student_id"] == "STU_TEST001"
    assert db_student["register_number"] == "TEST001"
    assert db_student["name"] == "Test Student"

    embs = db.get_embeddings_for_student(db_student["student_id"])
    assert len(embs) == 5

    # 2. Verify in FAISS Vector Store
    assert store.total_vectors == 5

    # 3. Simulate camera recognition pipeline:
    # Camera -> Face -> Embedding -> FAISS search -> student_id -> New Enrollment DB -> Register No + Name
    test_face_tensor = np.zeros((1, 3, 112, 112), dtype=np.float32)
    # Query with the exact first sample's aligned tensor
    first_sample_val = service.validate_frame(samples[0])
    assert first_sample_val.can_capture is True
    probe_tensor = first_sample_val.aligned_tensor

    recognizer = FaceRecognizer(embedder=embedder, vector_store=store, threshold=0.60)
    ready_face = RecognitionReadyFace(
        quality_metrics=FaceQualityResult(
            face_id="probe_test_01",
            bbox=[50, 50, 200, 250],
            keypoints=first_sample_val.quality_metrics.get("keypoints"),
            width=150,
            height=200,
            area=30000,
            sharpness=120.0,
            brightness=140.0,
            pose="FRONTAL",
            quality_status="RECOGNITION_READY"
        ),
        original_bbox=[50, 50, 200, 250],
        aligned_face_tensor=probe_tensor,
        timestamp=1000.0
    )

    rec_result = recognizer.recognize_face(ready_face)

    # Verify Recognition Match
    assert rec_result.status == RecognitionStatus.MATCH
    assert rec_result.matched_student_id == db_student["student_id"]
    assert rec_result.matched_student_name == "Test Student"

    # Query New Enrollment Database using matched_student_id to resolve Register Number + Name
    resolved_student = db.get_student_by_id(rec_result.matched_student_id)
    assert resolved_student is not None
    assert resolved_student["register_number"] == "TEST001"
    assert resolved_student["name"] == "Test Student"


# ---------------------------------------------------------------------------
# Test 12: REST API Endpoints for One-By-One Enrollment
# ---------------------------------------------------------------------------
def test_12_rest_api_one_by_one_endpoints(temp_env):
    import base64
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.state import get_app_state

    app = create_app()
    client = TestClient(app)
    state = get_app_state()

    # Use the isolated test database & store
    service = OneByOneEnrollmentService(
        db=temp_env["db"],
        vector_store=temp_env["store"],
        embedder=temp_env["embedder"],
        detector=MockDetector(mode="single"),
        storage_base_dir=temp_env["storage_dir"]
    )
    state._one_by_one_service = service

    # 1. Check duplicate on fresh number -> false
    resp = client.get("/api/enrollment/one-by-one/check/TEST_API_001")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["exists"] is False

    # 2. Validate sample frame via base64
    sample_img = make_synthetic_face(200, 250, brightness=140)
    _, buf = cv2.imencode(".jpg", sample_img)
    b64_str = base64.b64encode(buf).decode("utf-8")

    val_resp = client.post("/api/enrollment/one-by-one/validate-sample", json={"image_base64": b64_str})
    assert val_resp.status_code == 200
    val_data = val_resp.json()["data"]
    assert val_data["can_capture"] is True
    assert val_data["status"] == "PASS"
    assert "GOOD QUALITY" in val_data["quality_status"]
    assert val_data["crop_base64"] is not None

    # 3. Enroll student with 5 samples
    samples_b64 = []
    for i in range(5):
        s_img = make_synthetic_face(200, 250, brightness=130 + i * 10)
        _, s_buf = cv2.imencode(".jpg", s_img)
        samples_b64.append(base64.b64encode(s_buf).decode("utf-8"))

    enroll_payload = {
        "register_number": "TEST_API_001",
        "name": "API Student",
        "class_name": "III",
        "department": "AI&D",
        "section": "S-B",
        "samples": samples_b64
    }

    enroll_resp = client.post("/api/enrollment/one-by-one/enroll", json=enroll_payload)
    assert enroll_resp.status_code == 200
    enroll_data = enroll_resp.json()["data"]
    assert enroll_data["success"] is True
    assert enroll_data["register_number"] == "TEST_API_001"
    assert enroll_data["samples_enrolled"] == 5

    # 4. Check duplicate on enrolled number -> true
    check_dup = client.get("/api/enrollment/one-by-one/check/TEST_API_001")
    assert check_dup.status_code == 200
    assert check_dup.json()["data"]["exists"] is True

    # 5. List students
    list_resp = client.get("/api/enrollment/one-by-one/students")
    assert list_resp.status_code == 200
    students_list = list_resp.json()["data"]
    assert any(s["register_number"] == "TEST_API_001" for s in students_list)

    # 6. Get single student
    get_resp = client.get("/api/enrollment/one-by-one/student/TEST_API_001")
    assert get_resp.status_code == 200
    st_data = get_resp.json()["data"]
    assert st_data["name"] == "API Student"
    assert st_data["sample_count"] == 5

