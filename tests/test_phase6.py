import os
import sys
import time
import tempfile
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.face_embedder import ArcFaceEmbedder
from database.db_manager import DatabaseManager
from core.vector_store import FaissVectorStore
from core.recognizer import FaceRecognizer
from enrollment.enrollment_service import EnrollmentService
from core.schemas import RecognitionReadyFace, FaceQualityResult, RecognitionStatus, RecognitionResult


def create_test_face_tensor(seed: int = 42) -> np.ndarray:
    """Creates a deterministic synthetic normalized face tensor (1, 3, 112, 112) for testing."""
    rng = np.random.RandomState(seed)
    # Values roughly within [-1.0, 1.0] as preprocessed by ArcFace
    tensor = rng.uniform(-1.0, 1.0, size=(1, 3, 112, 112)).astype(np.float32)
    return tensor


@pytest.fixture
def temp_db_and_store():
    """Provides an isolated temporary SQLite database and FAISS vector store for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_smartclass.sqlite")
        index_path = os.path.join(tmpdir, "test_faiss.bin")
        db = DatabaseManager(db_path=db_path)
        store = FaissVectorStore(embedding_dim=512, index_path=index_path)
        yield db, store, tmpdir


# ---------------------------------------------------------------------------
# TEST 1: Model Loading
# ---------------------------------------------------------------------------
def test_01_model_loading():
    embedder = ArcFaceEmbedder()
    assert embedder.session is not None
    assert embedder.input_name == "input.1"
    assert embedder.output_shape[-1] == 512
    assert embedder.embedding_dim == 512

    # Verify exception on non-existent path
    with pytest.raises(FileNotFoundError):
        ArcFaceEmbedder(model_path="non_existent_weights.onnx")


# ---------------------------------------------------------------------------
# TEST 2: Valid Embedding Generation
# ---------------------------------------------------------------------------
def test_02_valid_embedding_generation():
    embedder = ArcFaceEmbedder()
    tensor = create_test_face_tensor(seed=101)
    embedding = embedder.generate_embedding(tensor)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 512)
    assert embedding.dtype == np.float32
    assert not np.isnan(embedding).any()
    assert not np.isinf(embedding).any()
    assert embedder.validate_embedding(embedding) is True


# ---------------------------------------------------------------------------
# TEST 3: Invalid Image Handling
# ---------------------------------------------------------------------------
def test_03_invalid_image_handling():
    embedder = ArcFaceEmbedder()

    # None input
    with pytest.raises(ValueError):
        embedder.generate_embedding(None)

    # Empty array
    with pytest.raises(ValueError):
        embedder.generate_embedding(np.array([], dtype=np.float32))

    # Incorrect spatial dimensions (e.g. 64x64 instead of 112x112)
    with pytest.raises(ValueError):
        embedder.generate_embedding(np.zeros((1, 3, 64, 64), dtype=np.float32))

    # Tensor containing NaN
    nan_tensor = np.zeros((1, 3, 112, 112), dtype=np.float32)
    nan_tensor[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        embedder.generate_embedding(nan_tensor)


# ---------------------------------------------------------------------------
# TEST 4: Embedding Dimension Validation
# ---------------------------------------------------------------------------
def test_04_embedding_dimension_validation():
    embedder = ArcFaceEmbedder()

    valid_emb = np.zeros((1, 512), dtype=np.float32)
    valid_emb[0, 0] = 1.0  # Unit norm
    assert embedder.validate_embedding(valid_emb) is True

    # 128-dim embedding -> must be rejected
    invalid_128 = np.zeros((1, 128), dtype=np.float32)
    invalid_128[0, 0] = 1.0
    assert embedder.validate_embedding(invalid_128) is False

    # 256-dim embedding -> must be rejected
    invalid_256 = np.zeros((1, 256), dtype=np.float32)
    invalid_256[0, 0] = 1.0
    assert embedder.validate_embedding(invalid_256) is False


# ---------------------------------------------------------------------------
# TEST 5: Embedding Normalization
# ---------------------------------------------------------------------------
def test_05_embedding_normalization():
    embedder = ArcFaceEmbedder()
    raw_vector = np.full((1, 512), 3.0, dtype=np.float32)
    assert abs(np.linalg.norm(raw_vector) - 1.0) > 0.1

    normalized = embedder.normalize_embedding(raw_vector)
    norm_val = float(np.linalg.norm(normalized))
    assert abs(norm_val - 1.0) < 1e-4

    # Output from generate_embedding must strictly have unit L2 norm
    tensor = create_test_face_tensor(seed=202)
    emb = embedder.generate_embedding(tensor)
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# TEST 6: Database Insertion
# ---------------------------------------------------------------------------
def test_06_database_insertion(temp_db_and_store):
    db, _, _ = temp_db_and_store
    db.add_student("TEST_001", "Alice Smith", department="AI & DS", section="A")

    student = db.get_student("TEST_001")
    assert student is not None
    assert student["student_name"] == "Alice Smith"

    vec = np.random.randn(512).astype(np.float32)
    vec = vec / np.linalg.norm(vec)

    emb_id = db.add_embedding("TEST_001", vec, quality_score=0.92, sample_label="frontal")
    assert emb_id > 0

    embs = db.get_embeddings_for_student("TEST_001")
    assert len(embs) == 1
    assert embs[0]["sample_label"] == "frontal"
    assert embs[0]["vector"].shape == (512,)
    assert abs(np.linalg.norm(embs[0]["vector"]) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# TEST 7: FAISS Insertion
# ---------------------------------------------------------------------------
def test_07_faiss_insertion(temp_db_and_store):
    _, store, _ = temp_db_and_store
    assert store.total_vectors == 0

    v = np.random.randn(512).astype(np.float32)
    v = v / np.linalg.norm(v)

    meta = {"student_id": "TEST_001", "student_name": "Alice"}
    store.add_vector(embedding_id=1, vector=v, metadata=meta)

    assert store.total_vectors == 1
    assert store.id_to_metadata[1]["student_id"] == "TEST_001"


# ---------------------------------------------------------------------------
# TEST 8: FAISS Search
# ---------------------------------------------------------------------------
def test_08_faiss_search(temp_db_and_store):
    _, store, _ = temp_db_and_store

    # Insert two orthogonal test vectors
    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(512, dtype=np.float32)
    v2[1] = 1.0

    store.add_vector(1, v1, {"student_id": "stud_A", "student_name": "Student A"})
    store.add_vector(2, v2, {"student_id": "stud_B", "student_name": "Student B"})

    # Search with exact v1
    res1 = store.search(v1, top_k=1)
    assert len(res1) == 1
    sim, meta = res1[0]
    assert abs(sim - 1.0) < 1e-4
    assert meta["student_id"] == "stud_A"

    # Search with exact v2
    res2 = store.search(v2, top_k=1)
    sim2, meta2 = res2[0]
    assert abs(sim2 - 1.0) < 1e-4
    assert meta2["student_id"] == "stud_B"


# ---------------------------------------------------------------------------
# TEST 9: Correct Test Identity Matching
# ---------------------------------------------------------------------------
def test_09_correct_test_identity_matching(temp_db_and_store):
    db, store, _ = temp_db_and_store
    embedder = ArcFaceEmbedder()
    enrollment = EnrollmentService(db, store, embedder)
    recognizer = FaceRecognizer(embedder=embedder, vector_store=store, threshold=0.65)

    # Enroll Test Student A
    enrollment.register_student("TEST_A", "Test Student A")
    tensor_a = create_test_face_tensor(seed=555)
    success, emb_id, _ = enrollment.enroll_face_sample("TEST_A", tensor_a, sample_label="frontal")
    assert success is True

    # Build RecognitionReadyFace for Test Student A
    ready_face_a = RecognitionReadyFace(
        quality_metrics=FaceQualityResult(
            face_id="query_face_a",
            bbox=[100, 100, 250, 250],
            width=150, height=150, area=22500,
            sharpness=120.0, brightness=110.0,
            quality_status="RECOGNITION_READY"
        ),
        original_bbox=[100, 100, 250, 250],
        aligned_face_tensor=tensor_a,
        timestamp=time.time()
    )

    result = recognizer.recognize_face(ready_face_a)
    assert result.status == RecognitionStatus.MATCH
    assert result.matched_student_id == "TEST_A"
    assert result.matched_student_name == "Test Student A"
    assert result.similarity >= 0.65
    assert result.similarity > 0.99  # Identical tensor yields ~1.0 similarity


# ---------------------------------------------------------------------------
# TEST 10: Unknown Rejection
# ---------------------------------------------------------------------------
def test_10_unknown_rejection(temp_db_and_store):
    db, store, _ = temp_db_and_store
    embedder = ArcFaceEmbedder()
    enrollment = EnrollmentService(db, store, embedder)
    # Using 0.85 threshold to separate synthetic noise impostors (~0.75-0.82) from genuine (1.00)
    recognizer = FaceRecognizer(embedder=embedder, vector_store=store, threshold=0.85)

    # Enroll Student A
    enrollment.register_student("TEST_A", "Student A")
    enrollment.enroll_face_sample("TEST_A", create_test_face_tensor(seed=1001))

    # Query with completely different face (Impostor)
    tensor_impostor = create_test_face_tensor(seed=9999)
    ready_face_imp = RecognitionReadyFace(
        quality_metrics=FaceQualityResult(
            face_id="impostor_face",
            bbox=[50, 50, 200, 200],
            width=150, height=150, area=22500,
            sharpness=100.0, brightness=100.0,
            quality_status="RECOGNITION_READY"
        ),
        original_bbox=[50, 50, 200, 200],
        aligned_face_tensor=tensor_impostor,
        timestamp=time.time()
    )

    result = recognizer.recognize_face(ready_face_imp)
    # Impostor should NOT be matched to Student A
    assert result.status == RecognitionStatus.UNKNOWN
    assert result.matched_student_id is None
    assert result.matched_student_name is None
    assert result.similarity < 0.85


# ---------------------------------------------------------------------------
# TEST 11: Threshold Behavior
# ---------------------------------------------------------------------------
def test_11_threshold_behavior(temp_db_and_store):
    _, store, _ = temp_db_and_store
    recognizer = FaceRecognizer(vector_store=store, threshold=0.65)

    # Add a mock embedding
    v = np.zeros(512, dtype=np.float32)
    v[0] = 1.0
    store.add_vector(1, v, {"student_id": "stud_1", "student_name": "Student 1"})

    # Create query vector that achieves cosine similarity ~0.707
    # angle 45 deg between axis 0 and axis 1
    q = np.zeros((1, 512), dtype=np.float32)
    q[0, 0] = 0.7071
    q[0, 1] = 0.7071

    # At threshold 0.65 -> similarity (0.707) >= 0.65 -> MATCH
    matches = store.search(q, top_k=1)
    sim, _ = matches[0]
    assert 0.70 < sim < 0.72

    recognizer.set_threshold(0.65)
    # Check manual threshold decision
    assert sim >= recognizer.threshold

    # Dynamically raise threshold to 0.85 -> similarity (0.707) < 0.85 -> UNKNOWN
    recognizer.set_threshold(0.85)
    assert sim < recognizer.threshold


# ---------------------------------------------------------------------------
# TEST 12: Multiple-Face Recognition
# ---------------------------------------------------------------------------
def test_12_multiple_face_recognition(temp_db_and_store):
    db, store, _ = temp_db_and_store
    embedder = ArcFaceEmbedder()
    enrollment = EnrollmentService(db, store, embedder)
    # Using 0.85 threshold to separate synthetic noise impostor (~0.76-0.79) from genuine (1.00)
    recognizer = FaceRecognizer(embedder=embedder, vector_store=store, threshold=0.85)

    # Enroll Alice (seed 1) and Bob (seed 2)
    enrollment.register_student("ID_ALICE", "Alice")
    enrollment.register_student("ID_BOB", "Bob")
    tensor_alice = create_test_face_tensor(seed=1)
    tensor_bob = create_test_face_tensor(seed=2)
    tensor_charlie_unknown = create_test_face_tensor(seed=3)

    enrollment.enroll_face_sample("ID_ALICE", tensor_alice)
    enrollment.enroll_face_sample("ID_BOB", tensor_bob)

    # Query 3 faces in single frame batch: Alice, Unknown Charlie, Bob
    f1 = RecognitionReadyFace(
        quality_metrics=FaceQualityResult(face_id="face_alice", bbox=[0,0,10,10], width=10, height=10, area=100, sharpness=100, brightness=100, quality_status="RECOGNITION_READY"),
        original_bbox=[0,0,10,10], aligned_face_tensor=tensor_alice, timestamp=1.0
    )
    f2 = RecognitionReadyFace(
        quality_metrics=FaceQualityResult(face_id="face_unknown", bbox=[10,10,20,20], width=10, height=10, area=100, sharpness=100, brightness=100, quality_status="RECOGNITION_READY"),
        original_bbox=[10,10,20,20], aligned_face_tensor=tensor_charlie_unknown, timestamp=1.0
    )
    f3 = RecognitionReadyFace(
        quality_metrics=FaceQualityResult(face_id="face_bob", bbox=[20,20,30,30], width=10, height=10, area=100, sharpness=100, brightness=100, quality_status="RECOGNITION_READY"),
        original_bbox=[20,20,30,30], aligned_face_tensor=tensor_bob, timestamp=1.0
    )

    results = recognizer.recognize_batch([f1, f2, f3])
    assert len(results) == 3

    assert results[0].status == RecognitionStatus.MATCH
    assert results[0].matched_student_id == "ID_ALICE"

    assert results[1].status == RecognitionStatus.UNKNOWN
    assert results[1].matched_student_id is None

    assert results[2].status == RecognitionStatus.MATCH
    assert results[2].matched_student_id == "ID_BOB"


# ---------------------------------------------------------------------------
# TEST 13: Empty Enrollment Database
# ---------------------------------------------------------------------------
def test_13_empty_enrollment_database(temp_db_and_store):
    _, store, _ = temp_db_and_store
    recognizer = FaceRecognizer(vector_store=store, threshold=0.65)
    assert store.total_vectors == 0

    tensor = create_test_face_tensor(seed=777)
    face = RecognitionReadyFace(
        quality_metrics=FaceQualityResult(face_id="f_empty", bbox=[0,0,10,10], width=10, height=10, area=100, sharpness=100, brightness=100, quality_status="RECOGNITION_READY"),
        original_bbox=[0,0,10,10], aligned_face_tensor=tensor, timestamp=1.0
    )

    result = recognizer.recognize_face(face)
    assert result.status == RecognitionStatus.UNKNOWN
    assert result.matched_student_id is None
    assert result.similarity == 0.0


# ---------------------------------------------------------------------------
# TEST 14: FAISS / Database Mismatch
# ---------------------------------------------------------------------------
def test_14_faiss_database_mismatch(temp_db_and_store):
    db, store, _ = temp_db_and_store
    db.add_student("S1", "Student 1")
    v = np.random.randn(512).astype(np.float32)
    v = v / np.linalg.norm(v)

    # Add 2 embeddings to SQLite
    db.add_embedding("S1", v)
    db.add_embedding("S1", v)
    assert db.get_embedding_count() == 2

    # But store has only 0 vectors
    is_valid, msg = store.validate_consistency(db.get_embedding_count())
    assert is_valid is False
    assert "FAISS / Database Mismatch" in msg

    # Synchronize
    all_embs = db.get_all_embeddings()
    store.sync_with_database(all_embs)

    is_valid2, _ = store.validate_consistency(db.get_embedding_count())
    assert is_valid2 is True
    assert store.total_vectors == 2


# ---------------------------------------------------------------------------
# TEST 15: Invalid Embedding
# ---------------------------------------------------------------------------
def test_15_invalid_embedding():
    embedder = ArcFaceEmbedder()

    assert embedder.validate_embedding(None) is False
    assert embedder.validate_embedding("invalid_string") is False
    assert embedder.validate_embedding(np.array([])) is False

    # Vector with wrong size
    assert embedder.validate_embedding(np.zeros(100, dtype=np.float32)) is False

    # Vector with NaN
    nan_v = np.zeros(512, dtype=np.float32)
    nan_v[0] = np.nan
    assert embedder.validate_embedding(nan_v) is False

    # Vector with Inf
    inf_v = np.zeros(512, dtype=np.float32)
    inf_v[0] = np.inf
    assert embedder.validate_embedding(inf_v) is False

    # Unnormalized vector (norm = 5.0)
    unnorm_v = np.full(512, 5.0, dtype=np.float32)
    assert embedder.validate_embedding(unnorm_v) is False


# ---------------------------------------------------------------------------
# TEST 16: Model Inference Failure / Robust Error Handling
# ---------------------------------------------------------------------------
def test_16_model_inference_failure(temp_db_and_store):
    _, store, _ = temp_db_and_store
    recognizer = FaceRecognizer(vector_store=store, threshold=0.65)

    # Face with None tensor
    broken_face = RecognitionReadyFace(
        quality_metrics=FaceQualityResult(face_id="broken", bbox=[0,0,10,10], width=10, height=10, area=100, sharpness=100, brightness=100, quality_status="RECOGNITION_READY"),
        original_bbox=[0,0,10,10], aligned_face_tensor=None, timestamp=1.0
    )

    res = recognizer.recognize_face(broken_face)
    assert res.status == RecognitionStatus.INVALID
    assert res.matched_student_id is None

    # Entire recognizer handles None input gracefully
    res_none = recognizer.recognize_face(None)
    assert res_none.status == RecognitionStatus.INVALID
