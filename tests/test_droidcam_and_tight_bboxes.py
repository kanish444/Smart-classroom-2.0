import cv2
import os
import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from camera.base_camera import BaseCamera
from camera.droidcam_camera import DroidCamCamera
from camera.camera_manager import CameraManager
from camera.smartboard_camera import SmartBoardCamera
from core.detector import YOLOv8FaceDetector
from core.face_processor import FaceProcessor
from core.recognizer import FaceRecognizer
from core.tracking_pipeline import TrackingPipeline
from attendance.session_manager import SessionManager
from attendance.attendance_engine import AttendanceEngine
from database.db_manager import DatabaseManager
from config.settings import get_settings
from app.main import create_app
from app.state import get_app_state, AppState
from core.schemas import SessionState


# =============================================================================
# PART 1: CAMERA TESTS (Scenarios 1 - 10)
# =============================================================================

def test_01_droidcam_url_reachable():
    """1. Test DroidCam base URL reachability probe logic."""
    reachable, msg = DroidCamCamera.probe_endpoint("127.0.0.1", 4747, timeout=0.05)
    # Even if offline, probe function must return a valid boolean and explanatory string
    assert isinstance(reachable, bool)
    assert isinstance(msg, str)
    assert len(msg) > 0


def test_02_droidcam_video_endpoint_reachable():
    """2. Test /video endpoint URL separation from remote-control base URL."""
    cam = DroidCamCamera(host="10.140.159.218", port=4747, video_path="/video")
    assert cam.base_url == "http://10.140.159.218:4747"
    assert cam.video_url == "http://10.140.159.218:4747/video"
    assert cam.base_url != cam.video_url
    assert "/video" in cam.video_url
    cam.release()


def test_03_valid_frame_received():
    """3. Test valid frame decoding, dimensions, and shape from camera."""
    cam = DroidCamCamera(host="10.140.159.218", port=4747)
    # Mock OpenCV capture returning a 720p valid frame
    fake_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, fake_frame)
    mock_cap.get.return_value = 30.0

    cam.cap = mock_cap
    cam.is_connected = True
    cam._actual_width = 1280
    cam._actual_height = 720

    success, frame = cam.get_frame()
    assert success is True
    assert frame is not None
    assert frame.shape == (720, 1280, 3)
    assert cam.resolution == (1280, 720)
    cam.release()


def test_04_invalid_ip():
    """4. Test probing an invalid / unreachable IP address returns controlled diagnostic."""
    reachable, msg = DroidCamCamera.probe_endpoint("192.0.2.1", 4747, timeout=0.1)
    assert not reachable
    assert any(term in msg.lower() for term in ["failed", "timed out", "refused", "connection"])


def test_05_droidcam_unavailable():
    """5. Test DroidCamCamera instantiation on offline host provides clear actionable reasons."""
    cam = DroidCamCamera(host="192.0.2.1", port=4747)
    assert not cam.is_connected
    assert "DroidCam video stream unavailable" in cam.last_error_message
    assert "Possible causes:" in cam.last_error_message
    assert "Wi-Fi" in cam.last_error_message
    assert "Firewall" in cam.last_error_message
    cam.release()


def test_06_reconnect_mechanism():
    """6. Test CameraManager controlled reconnect logic with retries and delay."""
    mgr = CameraManager(source="droidcam")
    # Patch _initialize_source so that the camera remains failing
    with patch.object(mgr, "_initialize_source") as mock_init:
        mock_cam = MagicMock()
        mock_cam.get_frame.return_value = (False, None)
        mock_cam.is_connected = False
        mgr.camera = mock_cam

        with patch("time.sleep", return_value=None):
            success, frame = mgr.get_frame(max_retries=2)
            assert success is False
            assert frame is None
            assert mock_init.call_count == 2
    mgr.release()



def test_07_laptop_camera():
    """7. Test laptop camera source initialization with configurable index."""
    mgr = CameraManager(source="laptop")
    assert mgr.current_source == "laptop"
    assert isinstance(mgr.camera, SmartBoardCamera)
    assert mgr.camera.camera_index == get_settings().camera.laptop_index
    mgr.release()


def test_08_camera_switching():
    """8. Test hot-swapping camera sources without restarting app."""
    mgr = CameraManager(source="laptop")
    assert mgr.current_source == "laptop"

    # Switch to DroidCam
    success, msg = mgr.switch_source("droidcam", host="10.140.159.218", port=4747)
    assert mgr.current_source == "droidcam"
    assert isinstance(mgr.camera, DroidCamCamera)
    assert mgr.camera.host == "10.140.159.218"

    # Switch to Smart Board
    success, msg = mgr.switch_source("smart_board", index=1)
    assert mgr.current_source == "smart_board"
    assert isinstance(mgr.camera, SmartBoardCamera)

    # Switch to External
    success, msg = mgr.switch_source("external", index=2)
    assert mgr.current_source == "external"
    assert isinstance(mgr.camera, SmartBoardCamera)

    # Switch back to Laptop
    success, msg = mgr.switch_source("laptop", index=0)
    assert mgr.current_source == "laptop"
    mgr.release()


def test_09_resolution_change():
    """9. Test resolution reporting in CameraManager diagnostic status."""
    mgr = CameraManager(source="droidcam")
    status = mgr.get_status()
    assert "resolution" in status
    assert "source_type" in status
    assert status["source_type"] == "droidcam"
    mgr.release()


def test_10_fps_measurement():
    """10. Test actual measured FPS calculation in DroidCamCamera."""
    cam = DroidCamCamera(host="10.140.159.218", port=4747)
    cam.is_connected = True
    cam._start_time = time.time() - 2.0  # 2 seconds elapsed
    cam._frame_count = 60               # 60 frames read
    measured_fps = cam.get_true_fps()
    assert 28.0 <= measured_fps <= 32.0
    cam.release()


# =============================================================================
# PART 2: DETECTION TESTS (Scenarios 11 - 21)
# =============================================================================

def test_11_one_face_tight_detection():
    """11. Test detector returns tight bounding box around a single detected face."""
    detector = YOLOv8FaceDetector()
    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)

    # Mock single face detection
    mock_box = MagicMock()
    mock_box.xyxy = [np.array([200, 150, 300, 270])]  # W: 100, H: 120
    mock_box.conf = [np.array(0.92)]
    mock_box.cls = [np.array(0)]
    mock_res = MagicMock()
    mock_res.boxes = [mock_box]
    mock_res.keypoints = None

    detector.model = MagicMock(return_value=[mock_res])
    dets = detector.detect(frame)

    assert len(dets) == 1
    x1, y1, x2, y2 = dets[0]["bbox"]
    assert x1 == 200 and y1 == 150 and x2 == 300 and y2 == 270
    assert (x2 - x1) == 100
    assert (y2 - y1) == 120
    assert dets[0]["crop_bbox"] == [200, 150, 300, 270]  # padding ratio = 0.0


def test_12_two_faces_separate_boxes():
    """12. Test two distinct faces receive two distinct non-overlapping boxes."""
    detector = YOLOv8FaceDetector()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    box1 = MagicMock(xyxy=[np.array([100, 150, 180, 250])], conf=[np.array(0.90)], cls=[np.array(0)])
    box2 = MagicMock(xyxy=[np.array([400, 150, 480, 250])], conf=[np.array(0.88)], cls=[np.array(0)])
    mock_res = MagicMock(boxes=[box1, box2], keypoints=None)
    detector.model = MagicMock(return_value=[mock_res])

    dets = detector.detect(frame)
    assert len(dets) == 2
    assert dets[0]["bbox"] != dets[1]["bbox"]
    # Check no overlap
    b1 = dets[0]["bbox"]
    b2 = dets[1]["bbox"]
    assert b1[2] < b2[0]  # Face 1 ends at 180, Face 2 starts at 400


def test_13_multiple_faces_separate_boxes():
    """13. Test 10+ faces each receive their own bounding box without merging."""
    detector = YOLOv8FaceDetector()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    boxes = []
    for i in range(12):
        bx1 = 50 + i * 90
        by1 = 200
        bx2 = bx1 + 70
        by2 = by1 + 90
        box = MagicMock(xyxy=[np.array([bx1, by1, bx2, by2])], conf=[np.array(0.85)], cls=[np.array(0)])
        boxes.append(box)

    mock_res = MagicMock(boxes=boxes, keypoints=None)
    detector.model = MagicMock(return_value=[mock_res])

    dets = detector.detect(frame)
    assert len(dets) == 12
    # Verify every face has a distinct box
    unique_boxes = {tuple(d["bbox"]) for d in dets}
    assert len(unique_boxes) == 12


def test_14_nearby_faces_no_merging():
    """14. Test nearby adjacent faces are not merged into one giant box."""
    detector = YOLOv8FaceDetector()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Two students sitting next to each other
    box_left = MagicMock(xyxy=[np.array([200, 150, 280, 250])], conf=[np.array(0.91)], cls=[np.array(0)])
    box_right = MagicMock(xyxy=[np.array([290, 150, 370, 250])], conf=[np.array(0.89)], cls=[np.array(0)])
    mock_res = MagicMock(boxes=[box_left, box_right], keypoints=None)
    detector.model = MagicMock(return_value=[mock_res])

    dets = detector.detect(frame)
    assert len(dets) == 2
    assert dets[0]["bbox"] == [200, 150, 280, 250]
    assert dets[1]["bbox"] == [290, 150, 370, 250]
    # Width of each box should be 80, not 170 (merged)
    assert (dets[0]["bbox"][2] - dets[0]["bbox"][0]) == 80
    assert (dets[1]["bbox"][2] - dets[1]["bbox"][0]) == 80


def test_15_distant_faces_quality_separation():
    """15. Test distant small faces are categorized in FAR zone and separated from recognition quality."""
    detector = YOLOv8FaceDetector()
    # Face area = 40 * 50 = 2000 px (< 3000 FAR threshold)
    zone = detector.get_zone(area=2000, frame_width=1280, frame_height=720)
    assert zone == "FAR"

    # Middle zone: 5000 px
    assert detector.get_zone(area=5000, frame_width=1280, frame_height=720) == "MIDDLE"

    # Near zone: 20000 px
    assert detector.get_zone(area=20000, frame_width=1280, frame_height=720) == "NEAR"


def test_16_small_face_rejection():
    """16. Test tiny artifact boxes smaller than min_face_size are rejected."""
    detector = YOLOv8FaceDetector()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    good_box = MagicMock(xyxy=[np.array([200, 200, 280, 300])], conf=[np.array(0.90)], cls=[np.array(0)])
    tiny_box = MagicMock(xyxy=[np.array([10, 10, 18, 18])], conf=[np.array(0.50)], cls=[np.array(0)]) # 8x8 < 20px
    mock_res = MagicMock(boxes=[good_box, tiny_box], keypoints=None)
    detector.model = MagicMock(return_value=[mock_res])

    dets = detector.detect(frame)
    assert len(dets) == 1
    assert dets[0]["bbox"] == [200, 200, 280, 300]


def test_17_edge_faces_clamping():
    """17. Test faces at frame edges are clamped to 0 <= x1 < x2 <= w, 0 <= y1 < y2 <= h."""
    detector = YOLOv8FaceDetector()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Box spilling out on top and left: raw [-20, -15, 80, 100]
    edge_box = MagicMock(xyxy=[np.array([-20, -15, 80, 100])], conf=[np.array(0.85)], cls=[np.array(0)])
    # Box spilling out on bottom and right: raw [1200, 680, 1350, 780]
    edge_box2 = MagicMock(xyxy=[np.array([1200, 680, 1350, 780])], conf=[np.array(0.82)], cls=[np.array(0)])
    mock_res = MagicMock(boxes=[edge_box, edge_box2], keypoints=None)
    detector.model = MagicMock(return_value=[mock_res])

    dets = detector.detect(frame)
    assert len(dets) == 2
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 1280
        assert y2 <= 720
        assert x1 < x2
        assert y1 < y2


def test_18_duplicate_detections_nms():
    """18. Test NMS parameter configuration prevents duplicate detections."""
    settings = get_settings().detection
    assert settings.nms_iou_threshold == 0.45
    assert settings.confidence_threshold == 0.35
    assert settings.min_face_size == 20


def test_19_invalid_inverted_coordinates_rejected():
    """19. Test degenerate/inverted boxes (x2 <= x1 or y2 <= y1) are safely rejected."""
    detector = YOLOv8FaceDetector()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    inv_box1 = MagicMock(xyxy=[np.array([200, 200, 180, 250])], conf=[np.array(0.80)], cls=[np.array(0)]) # x2 < x1
    inv_box2 = MagicMock(xyxy=[np.array([200, 250, 280, 200])], conf=[np.array(0.80)], cls=[np.array(0)]) # y2 < y1
    mock_res = MagicMock(boxes=[inv_box1, inv_box2], keypoints=None)
    detector.model = MagicMock(return_value=[mock_res])

    dets = detector.detect(frame)
    assert len(dets) == 0


def test_20_letterboxed_frame_coordinates():
    """20. Test detector handles letterboxed or non-standard aspect ratio frames without crashing."""
    detector = YOLOv8FaceDetector()
    # Wide cinematic aspect ratio
    frame = np.zeros((540, 1920, 3), dtype=np.uint8)
    dets = detector.detect(frame)
    assert isinstance(dets, list)


def test_21_resized_frame_coordinates():
    """21. Test detector handles lower resolution resized frames (e.g. 640x360)."""
    detector = YOLOv8FaceDetector()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    dets = detector.detect(frame)
    assert isinstance(dets, list)


# =============================================================================
# PART 3: PIPELINE & INTEGRATION TESTS (Scenarios 22 - 28)
# =============================================================================

def test_22_droidcam_plus_yolo_pipeline():
    """22. Test end-to-end integration: DroidCam frame into YOLO detector."""
    cam = DroidCamCamera(host="10.140.159.218", port=4747)
    fake_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, fake_frame)
    cam.cap = mock_cap
    cam.is_connected = True

    detector = YOLOv8FaceDetector()
    mock_box = MagicMock(xyxy=[np.array([300, 200, 400, 320])], conf=[np.array(0.91)], cls=[np.array(0)])
    detector.model = MagicMock(return_value=[MagicMock(boxes=[mock_box], keypoints=None)])

    success, frame = cam.get_frame()
    assert success is True
    dets = detector.detect(frame)
    assert len(dets) == 1
    assert dets[0]["bbox"] == [300, 200, 400, 320]
    cam.release()


def test_23_droidcam_plus_recognition():
    """23. Test DroidCam frame through YOLO detector and FaceProcessor to ArcFace."""
    pipeline = TrackingPipeline()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    detections = [{
        "bbox": [200, 150, 290, 270],
        "confidence": 0.90,
        "face_area": 90 * 120,
        "zone": "NEAR",
        "keypoints": [[225, 190], [265, 190], [245, 215], [230, 245], [260, 245]]
    }]
    tracks = pipeline.process_frame(frame, detections=detections)
    assert len(tracks) == 1
    assert tracks[0].bbox == [200, 150, 290, 270]


def test_24_droidcam_plus_bytetrack():
    """24. Test DroidCam frame tracked by ByteTrack preserves tight bounding box dimensions."""
    pipeline = TrackingPipeline()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    detections = [{
        "bbox": [350, 180, 430, 280],  # W: 80, H: 100
        "confidence": 0.89,
        "face_area": 80 * 100,
        "zone": "NEAR",
        "keypoints": [[370, 210], [410, 210], [390, 230], [380, 255], [405, 255]]
    }]
    tracks = pipeline.process_frame(frame, detections=detections)
    assert len(tracks) == 1
    bx1, by1, bx2, by2 = tracks[0].bbox
    bw = bx2 - bx1
    bh = by2 - by1
    assert bw <= 90, f"Tracked width {bw} should tightly match detected face width (80)"
    assert bh <= 110, f"Tracked height {bh} should tightly match detected face height (100)"


def test_25_droidcam_plus_attendance():
    """25. Test AttendanceEngine processes tracked faces without duplicating attendance."""
    db = DatabaseManager()
    sess_mgr = SessionManager(db_manager=db)
    engine = AttendanceEngine(db_manager=db, session_manager=sess_mgr)

    # Seed student and session
    sid = f"STU_DTEST_{int(time.time())}"
    db.add_student(sid, "Test DroidCam Student", "CSE", "A")
    sess_id = f"SESS_DTEST_{int(time.time()*1000)}"
    sec_name = f"CSE_TEST_{int(time.time()*1000)}"
    sess = sess_mgr.create_session(sess_id, "2026-09-24", sec_name, "AI Lab", "09:00", "10:00")
    sess_mgr.start_session(sess.session_id)

    try:
        # Simulate tracked student observed
        from core.schemas import TrackedFace, TrackState, RecognitionStatus
        face = TrackedFace(
            track_id=1,
            bbox=[200, 150, 290, 270],
            state=TrackState.ACTIVE,
            score=0.92,
            stable_student_id=sid,
            stable_student_name="Test DroidCam Student",
            current_status=RecognitionStatus.MATCH,
            current_similarity=0.88,
            quality_status="RECOGNITION_READY"
        )

        # Process first observation -> marked present
        res1 = engine.process_tracked_faces(sess.session_id, [face])
        # Process second observation in same session -> deduped, not duplicated
        res2 = engine.process_tracked_faces(sess.session_id, [face])

        records = db.get_attendance_for_session(sess.session_id)
        student_records = [r for r in records if r["student_id"] == sid]
        assert len(student_records) == 1, "Must have exactly 1 attendance record (no duplicates)"
    finally:
        sess_mgr.end_session(sess.session_id)


def test_26_droidcam_plus_dashboard():
    """26. Test dashboard feeds support normal, debug, and raw views."""
    app = create_app()
    client = TestClient(app)

    # Test /api/video/feed with normal view
    res = client.get("/api/video/feed?view_mode=normal&limit=1")
    assert res.status_code == 200
    assert "multipart/x-mixed-replace" in res.headers.get("content-type", "")

    # Test /api/video/feed with debug view
    res = client.get("/api/video/feed?view_mode=debug&limit=1")
    assert res.status_code == 200

    # Test /api/video/feed with raw detector view
    res = client.get("/api/video/feed?view_mode=raw&limit=1")
    assert res.status_code == 200




def test_27_camera_disconnect_during_session():
    """27. Test camera disconnect generates placeholder frame and does not crash application."""
    state = get_app_state()
    state.camera_online = False
    disconn_frame = state._generate_disconnected_frame(
        source_name="droidcam",
        target_desc="http://10.140.159.218:4747/video",
        reason="Wi-Fi connection lost"
    )
    assert disconn_frame is not None
    assert disconn_frame.shape == (720, 1280, 3)

    # Rendered frame during disconnect
    rendered = state.get_rendered_frame(view_mode="normal")
    assert rendered is not None
    assert rendered.shape[:2] == (720, 1280)


def test_28_camera_reconnect_during_session():
    """28. Test camera reconnect restores stream and online status gracefully."""
    state = get_app_state()
    # Simulate reconnect
    valid_frame = np.full((720, 1280, 3), 150, dtype=np.uint8)
    state.update_telemetry(frame=valid_frame, tracks=[], fps=30.0)
    state.camera_online = True

    rendered = state.get_rendered_frame(view_mode="normal")
    assert rendered is not None
    assert state.camera_online is True
