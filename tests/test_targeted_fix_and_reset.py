import os
import json
import sqlite3
import numpy as np
import pytest
from unittest.mock import MagicMock

from config.settings import get_settings
from core.detector import YOLOv8FaceDetector
from core.vector_store import FaissVectorStore
from database.new_enrollment_db import NewEnrollmentDatabase
from database.db_manager import DatabaseManager
from app.state import get_app_state
from core.schemas import TrackedFace, TrackState, RecognitionStatus


def test_01_face_detection_works_and_tight():
    """TEST 1 & 2: Face detection still works and bounding boxes are correctly localized & clamped."""
    detector = YOLOv8FaceDetector()
    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)

    # Mock ultralytics detection
    mock_box = MagicMock(
        xyxy=[np.array([300.2, 150.8, 480.6, 380.1])],
        conf=[np.array(0.92)],
        cls=[np.array(0)]
    )
    mock_res = MagicMock(boxes=[mock_box], keypoints=None)
    detector.model = MagicMock(return_value=[mock_res])

    dets = detector.detect(frame)
    assert len(dets) == 1
    # Check rounding and tight bbox
    assert dets[0]["bbox"] == [300, 151, 481, 380]
    assert dets[0]["confidence"] >= 0.90
    assert len(dets[0]["keypoints"]) == 5
    # Check aspect ratio
    w = dets[0]["bbox"][2] - dets[0]["bbox"][0]
    h = dets[0]["bbox"][3] - dets[0]["bbox"][1]
    aspect = w / float(h)
    assert 0.4 <= aspect <= 2.0


def test_02_malformed_and_duplicate_rejection():
    """TEST 2: Face detector rejects malformed slivers and duplicate overlapping boxes."""
    detector = YOLOv8FaceDetector()
    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)

    # 1. Valid face box
    box_good = MagicMock(xyxy=[np.array([100, 100, 200, 240])], conf=[np.array(0.85)], cls=[np.array(0)])
    # 2. Duplicate box on same face with high IoU
    box_dup = MagicMock(xyxy=[np.array([105, 105, 202, 242])], conf=[np.array(0.65)], cls=[np.array(0)])
    # 3. Malformed narrow vertical sliver (aspect < 0.4)
    box_sliver = MagicMock(xyxy=[np.array([300, 100, 315, 300])], conf=[np.array(0.80)], cls=[np.array(0)])
    # 4. Inverted box
    box_inv = MagicMock(xyxy=[np.array([400, 200, 350, 250])], conf=[np.array(0.80)], cls=[np.array(0)])

    mock_res = MagicMock(boxes=[box_good, box_dup, box_sliver, box_inv], keypoints=None)
    detector.model = MagicMock(return_value=[mock_res])

    dets = detector.detect(frame)
    assert len(dets) == 1
    assert dets[0]["bbox"] == [100, 100, 200, 240]


def test_03_no_old_students_remain():
    """TEST 3 & 4: Enrollment database count = 0 and no old students remain."""
    new_db = NewEnrollmentDatabase()
    assert new_db.get_student_count() == 0
    assert new_db.get_embedding_count() == 0
    assert len(new_db.get_all_students()) == 0

    main_db = DatabaseManager()
    assert main_db.get_student_count() == 0
    assert main_db.get_embedding_count() == 0
    assert len(main_db.get_all_students()) == 0


def test_05_faiss_vector_count_zero():
    """TEST 5: FAISS vector count = 0 and metadata mappings = 0."""
    v_store = FaissVectorStore()
    assert v_store.total_vectors == 0
    assert len(v_store.id_to_metadata) == 0

    # Ensure search on empty store returns [] safely without crashing
    query = np.ones((512,), dtype=np.float32)
    matches = v_store.search(query, top_k=5)
    assert matches == []


def test_06_enrollment_photos_zero():
    """TEST 6: Stored enrollment photos = 0."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    enroll_dir = os.path.join(base_dir, "data", "enrollment")
    if os.path.exists(enroll_dir):
        student_dirs = [d for d in os.listdir(enroll_dir) if os.path.isdir(os.path.join(enroll_dir, d))]
        assert len(student_dirs) == 0
        img_files = [f for f in os.listdir(enroll_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        assert len(img_files) == 0


def test_07_detected_face_shows_unknown():
    """TEST 7: With zero enrolled students, any detected face shows UNKNOWN."""
    state = get_app_state()
    state.reset_state()

    # Verify vector store has 0 vectors
    v_store = state.get_vector_store()
    v_store.clear()
    assert v_store.total_vectors == 0

    # Create dummy frame and tracked face with no enrolled student
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    unregistered_track = TrackedFace(
        track_id=1,
        bbox=[300, 200, 450, 400],
        state=TrackState.ACTIVE,
        score=0.90,
        stable_student_id=None,
        stable_student_name=None,
        current_status=RecognitionStatus.UNKNOWN,
        current_similarity=0.12,
        quality_status="RECOGNITION_READY"
    )

    rendered = state._render_normal_view(frame, [unregistered_track])
    assert rendered is not None
    assert rendered.shape == frame.shape
    # Ensure telemetry reports unknown
    state.update_telemetry(tracks=[unregistered_track])
    telem = state.get_latest_telemetry()
    assert telem["unknown_count"] == 1
    assert telem["tracks"][0].stable_student_id is None


def test_08_add_new_data_and_class_dropdown():
    """TEST 8 & 9: + ADD NEW DATA modal exists and Class dropdown contains 1st, 2nd, 3rd, 4th Year."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(base_dir, "dashboard", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Test + ADD NEW DATA button / modal trigger exists
    assert "btn-open-enrollment" in html
    assert "enrollment-modal" in html

    # Test Class select dropdown contains 1st Year, 2nd Year, 3rd Year, 4th Year
    assert '<select id="enroll-input-class"' in html
    assert '<option value="1st Year">1st Year</option>' in html
    assert '<option value="2nd Year">2nd Year</option>' in html
    assert 'value="3rd Year"' in html
    assert '3rd Year' in html
    assert '<option value="4th Year">4th Year</option>' in html
