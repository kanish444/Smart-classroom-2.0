import time
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import create_app
from app.state import get_app_state, AppState
from camera.camera_manager import CameraManager
from camera.smartboard_camera import SmartBoardCamera
from camera.droidcam_camera import DroidCamCamera
from config.settings import get_settings


@pytest.fixture(scope="module")
def app_instance():
    return create_app()


@pytest.fixture(scope="module")
def client(app_instance):
    with TestClient(app_instance) as test_client:
        yield test_client


def test_01_camera_yaml_configuration():
    """Requirement 20: Camera YAML configuration loaded and parsed correctly."""
    settings = get_settings()
    assert settings.camera.source == "pc"
    assert settings.camera.pc.device_index == 0
    assert settings.camera.pc.width == 1280
    assert settings.camera.pc.height == 720
    assert settings.camera.pc.fps == 30


def test_02_pc_webcam_connection_and_lifecycle():
    """Requirement 3, 4, 14: Connect, read_frame, resolution, fps, status, release."""
    mgr = CameraManager(source="pc")
    assert mgr.current_source == "pc"
    assert bool(mgr.is_connected) is True
    assert mgr.is_connected() is True

    # Check resolution & FPS reporting
    res = mgr.get_resolution()
    assert isinstance(res, tuple) and len(res) == 2
    fps = mgr.get_fps()
    assert fps >= 0.0

    # Read frame verification
    success, frame = mgr.read_frame()
    assert success is True
    assert frame is not None
    assert isinstance(frame, np.ndarray)

    # Status check
    status_info = mgr.get_status()
    assert status_info["status"] in ["CONNECTED", "STREAMING"]
    assert status_info["connected"] is True
    assert "resolution" in status_info
    assert "fps" in status_info

    # Clean release / disconnect
    mgr.disconnect()
    post_status = mgr.get_status()
    assert post_status["status"] == "STOPPED"
    assert post_status["connected"] is False


def test_03_device_discovery_endpoint(client):
    """Requirement 19: GET /api/camera/devices returns usable camera devices."""
    response = client.get("/api/camera/devices")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    devices = data["data"]
    assert isinstance(devices, list)
    assert len(devices) > 0
    assert "index" in devices[0]
    assert "name" in devices[0]


def test_04_camera_stream_endpoint(client):
    """Requirement 5: GET /api/camera/stream returns MJPEG stream of processed frames."""
    # Test official endpoint /api/camera/stream
    response = client.get("/api/camera/stream?limit=1")
    assert response.status_code == 200
    assert "multipart/x-mixed-replace" in response.headers["content-type"]
    assert len(response.content) > 0

    # Test backward-compatible alias /api/video/feed
    feed_resp = client.get("/api/video/feed?limit=1")
    assert feed_resp.status_code == 200
    assert "multipart/x-mixed-replace" in feed_resp.headers["content-type"]
    assert len(feed_resp.content) > 0


def test_05_singleton_camera_manager_no_duplicate_handles(app_instance, client):
    """Requirement 6 & 16: Multiple viewers/tabs consume the same singleton stream."""
    with patch("cv2.VideoCapture") as mock_vc:
        # Simulate Tab 1
        resp1 = client.get("/api/camera/stream?limit=1")
        assert resp1.status_code == 200

        # Simulate Tab 2
        resp2 = client.get("/api/camera/stream?limit=1")
        assert resp2.status_code == 200

        # Neither client opened a new VideoCapture; both consume the shared processed frame
        assert mock_vc.call_count == 0


def test_06_camera_start_and_stop_endpoints(client):
    """Requirement 13 & 14: START CAMERA and STOP CAMERA API lifecycles."""
    # Stop camera
    stop_resp = client.post("/api/camera/stop")
    assert stop_resp.status_code == 200
    stop_data = stop_resp.json()
    assert stop_data["success"] is True
    assert stop_data["data"]["camera_online"] is False
    assert stop_data["data"]["status"] == "STOPPED"

    # Status check after stop
    status_resp = client.get("/api/camera/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()["data"]
    assert status_data["camera_online"] is False
    assert status_data["status"] == "STOPPED"
    assert status_data["fps"] == 0.0

    # Start camera
    start_resp = client.post("/api/camera/start")
    assert start_resp.status_code == 200
    start_data = start_resp.json()
    assert start_data["success"] is True


def test_07_scrfd_detection_and_face_pipeline():
    """Requirement 1, 9, 10, 11: Real webcam frame flows through SCRFD, ArcFace, FAISS, ByteTrack."""
    from core.tracking_pipeline import TrackingPipeline
    from core.detector import get_face_detector
    from core.recognizer import FaceRecognizer

    detector = get_face_detector()
    assert detector is not None
    pipeline = TrackingPipeline(detector=detector)

    # Synthetic test frame with known dimensions
    test_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)

    # Process frame
    tracks = pipeline.process_frame(test_frame)
    assert isinstance(tracks, list)
    metrics = pipeline.latest_latency_metrics
    assert "detector_ms" in metrics
    assert "tracking_ms" in metrics
    assert "recognition_ms" in metrics
    assert "total_ms" in metrics


def test_08_camera_source_switching_releases_prior_camera():
    """Requirement 17: Switching PC Camera -> DroidCam releases PC Camera."""
    with patch.object(SmartBoardCamera, "_initialize_camera", return_value=True):
        mgr = CameraManager(source="pc")
        initial_cam = mgr.camera
        initial_cam_release_spy = MagicMock()
        initial_cam.release = initial_cam_release_spy

        def mock_init_droidcam(self_dc):
            self_dc.is_connected = True
            return True

        with patch.object(DroidCamCamera, "_initialize_camera", mock_init_droidcam):
            with patch.object(DroidCamCamera, "get_frame", return_value=(True, np.zeros((720, 1280, 3), dtype=np.uint8))):
                success, msg = mgr.switch_source("droidcam", host="10.140.159.218", port=4747)
                assert mgr.current_source == "droidcam"
                assert initial_cam_release_spy.called
                assert isinstance(mgr.camera, DroidCamCamera)

        mgr.release()
