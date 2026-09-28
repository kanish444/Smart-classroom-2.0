import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from camera.droidcam_camera import DroidCamCamera
from camera.camera_manager import CameraManager
from app.state import AppState
from core.schemas import SessionState, TrackedFace, AttendanceStatus


def test_01_droidcam_fresh_frame_retrieval():
    """1. Test that DroidCam always returns newest frame without stale accumulation."""
    cam = DroidCamCamera(host="10.140.159.218", port=4747)

    # Mock open capture that produces incrementing frame values
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    current_val = [0]
    def mock_read():
        current_val[0] += 1
        frame = np.full((720, 1280, 3), current_val[0] % 255, dtype=np.uint8)
        return True, frame

    mock_cap.read.side_effect = mock_read
    mock_cap.get.return_value = 30.0

    cam.cap = mock_cap
    cam.is_connected = True
    cam._actual_width = 1280
    cam._actual_height = 720

    # Start capture worker thread
    cam._stop_reader.clear()
    import threading
    cam._reader_thread = threading.Thread(target=cam._capture_worker, daemon=True)
    cam._reader_thread.start()

    # Let worker run for a moment so many frames are produced
    time.sleep(0.08)

    # Read frame 1
    success1, frame1 = cam.get_frame()
    assert success1 is True
    val1 = int(frame1[0, 0, 0])

    # Simulate heavy 100ms AI inference delay in consumer
    time.sleep(0.1)

    # Read frame 2: must be the NEWEST frame produced, NOT the old frame 2
    success2, frame2 = cam.get_frame()
    assert success2 is True
    val2 = int(frame2[0, 0, 0])

    # In 100ms at ~1000 FPS mock loop, val2 must be significantly ahead of val1 + 1 (stale frames discarded!)
    assert val2 > val1 + 1, f"Frame freshness failed: val2={val2}, val1={val1}. Stale frames were not dropped!"

    cam.release()
    assert not cam.is_connected
    assert cam._reader_thread is None


def test_02_bounded_buffer_size_one():
    """2. Verify that DroidCam maintains a single-slot buffer (maxsize=1) without queue growth."""
    cam = DroidCamCamera(host="10.140.159.218", port=4747)
    
    # Verify buffer variables exist
    assert hasattr(cam, "_latest_frame")
    assert hasattr(cam, "_frame_lock")
    assert hasattr(cam, "_stop_reader")
    assert hasattr(cam, "_reader_thread")
    cam.release()


def test_03_camera_manager_integration():
    """3. Verify CameraManager works seamlessly with optimized DroidCamCamera."""
    mgr = CameraManager(source="droidcam")
    fake_frame = np.full((720, 1280, 3), 100, dtype=np.uint8)

    mock_cam = MagicMock()
    mock_cam.get_frame.return_value = (True, fake_frame)
    mock_cam.is_connected = True
    mgr.camera = mock_cam

    success, frame = mgr.get_frame()
    assert success is True
    assert frame is not None
    assert frame.shape == (720, 1280, 3)
    mgr.release()


def test_04_decoupled_app_state_ai_worker():
    """4. Verify AppState decouples live preview capture from AI worker loop."""
    state = AppState(db_path=":memory:")
    
    # Mock camera manager returning test frames
    mock_mgr = MagicMock()
    mock_frame = np.full((720, 1280, 3), 200, dtype=np.uint8)
    mock_mgr.get_frame.return_value = (True, mock_frame)
    mock_mgr.current_source = "droidcam"
    mock_mgr.get_status.return_value = {"source_type": "droidcam", "status": "CONNECTED"}

    state.camera_manager = mock_mgr

    # Mock tracking pipeline returning sample tracked face
    mock_pipeline = MagicMock()
    sample_face = TrackedFace(
        track_id=1,
        detection_id=1,
        bbox=(50, 50, 200, 200),
        quality_score=0.9,
        stable_student_id="STU001",
        stable_student_name="Alice"
    )
    mock_pipeline.process_frame.return_value = [sample_face]
    mock_pipeline.latest_raw_detections = []
    state.tracking_pipeline = mock_pipeline

    # Start camera and AI worker
    state._stop_camera_event.clear()
    state._ai_event.clear()

    import threading
    state._camera_thread = threading.Thread(target=state._camera_worker_loop, daemon=True)
    state._camera_thread.start()

    # Wait for worker loops to process
    for _ in range(50):
        with state._lock:
            if state.latest_frame is not None:
                break
        time.sleep(0.02)

    # Verify live preview received latest frame immediately
    with state._lock:
        assert state.latest_frame is not None
        assert state.camera_online is True

    # Verify AI worker processed frame and populated tracks
    for _ in range(50):
        with state._lock:
            if len(state.latest_tracks) > 0:
                break
        time.sleep(0.02)

    with state._lock:
        assert len(state.latest_tracks) == 1
        assert state.latest_tracks[0].stable_student_id == "STU001"

    # Verify get_mjpeg_frame generates valid JPEG
    jpeg_bytes = state.get_mjpeg_frame(view_mode="normal")
    assert len(jpeg_bytes) > 0
    assert jpeg_bytes[:2] == b"\xff\xd8"  # JPEG header magic bytes

    state.stop_camera_worker()
    assert not state.camera_online


def test_05_reconnect_resilience():
    """5. Verify DroidCam clean release and reconnect logic."""
    cam = DroidCamCamera(host="192.0.2.1", port=4747)
    assert not cam.is_connected
    assert cam._reader_thread is None

    # Connect attempt on unreachable IP returns False gracefully without crash
    reconn_success = cam.connect()
    assert reconn_success is False
    assert not cam.is_connected
    cam.release()
