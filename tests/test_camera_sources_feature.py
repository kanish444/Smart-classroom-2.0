import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import create_app
from camera.camera_manager import CameraManager
from camera.smartboard_camera import SmartBoardCamera
from camera.droidcam_camera import DroidCamCamera
from camera.esp32_camera import ESP32Camera
from config.settings import get_settings


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_01_dashboard_loads(client):
    """1. Dashboard loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert "SmartClass Vision AI" in response.text


def test_02_camera_source_buttons_appear(client):
    """2. Camera source buttons appear on tracking dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    text = response.text
    assert "CAMERA SOURCE" in text
    assert "PC CAMERA" in text
    assert "DROIDCAM" in text
    assert "EXTENSION" in text
    assert "ESP32 CAMERA" in text


def test_03_pc_camera_selection(client):
    """3. PC camera selection works via API."""
    with patch.object(SmartBoardCamera, "_initialize_camera", return_value=True):
        response = client.post("/api/camera/select", json={"source": "pc", "index": 0})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["source"] == "pc"


def test_04_droidcam_configuration(client):
    """4. DroidCam configuration appears and endpoint probe works."""
    response = client.get("/api/camera/sources")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    d = data["data"]
    assert "droidcam_config" in d
    assert "host" in d["droidcam_config"]
    assert "port" in d["droidcam_config"]
    assert "video_url" in d["droidcam_config"]
    assert d["droidcam_config"]["host"] == "10.140.159.218"
    assert d["droidcam_config"]["port"] == 4747


def test_05_extension_camera_selection(client):
    """5. Extension camera selection and device enumeration appear."""
    response = client.get("/api/camera/sources")
    assert response.status_code == 200
    data = response.json()
    assert "available_devices" in data["data"]
    assert len(data["data"]["available_devices"]) > 0

    with patch.object(SmartBoardCamera, "_initialize_camera", return_value=True):
        resp = client.post("/api/camera/select", json={"source": "extension", "index": 1})
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert resp.json()["data"]["source"] == "extension"


def test_06_esp32_url_configuration(client):
    """6. ESP32 URL configuration appears and is configurable."""
    response = client.get("/api/camera/sources")
    assert response.status_code == 200
    data = response.json()
    assert "esp32_config" in data["data"]
    assert "stream_url" in data["data"]["esp32_config"]

    with patch.object(ESP32Camera, "probe_endpoint", return_value=(True, "Reachable")):
        with patch.object(ESP32Camera, "_initialize_camera", return_value=True):
            resp = client.post(
                "/api/camera/select",
                json={"source": "esp32", "stream_url": "http://192.168.1.100:81/stream"}
            )
            assert resp.status_code == 200
            assert resp.json()["success"] is True
            assert resp.json()["data"]["source"] == "esp32"


def test_07_camera_manager_accepts_selected_source():
    """7. CameraManager accepts all 4 sources and provides consistent frame interface."""
    # PC
    with patch.object(SmartBoardCamera, "_initialize_camera", return_value=True):
        mgr = CameraManager(source="pc")
        assert mgr.current_source == "pc"
        assert isinstance(mgr.camera, SmartBoardCamera)
        assert hasattr(mgr, "connect")
        assert hasattr(mgr, "read_frame")
        assert hasattr(mgr, "disconnect")
        assert hasattr(mgr, "is_connected")

        # Switch to DroidCam
        with patch.object(DroidCamCamera, "_initialize_camera", return_value=True):
            success, msg = mgr.switch_source("droidcam", host="10.140.159.218", port=4747)
            assert mgr.current_source == "droidcam"
            assert isinstance(mgr.camera, DroidCamCamera)

        # Switch to Extension
        with patch.object(SmartBoardCamera, "_initialize_camera", return_value=True):
            success, msg = mgr.switch_source("extension", index=1)
            assert mgr.current_source == "extension"
            assert isinstance(mgr.camera, SmartBoardCamera)

        # Switch to ESP32
        with patch.object(ESP32Camera, "_initialize_camera", return_value=True):
            success, msg = mgr.switch_source("esp32", stream_url="http://192.168.1.100:81/stream")
            assert mgr.current_source == "esp32"
            assert isinstance(mgr.camera, ESP32Camera)

        # Switch back to PC
        success, msg = mgr.switch_source("pc", index=0)
        assert mgr.current_source == "pc"
        assert isinstance(mgr.camera, SmartBoardCamera)

        mgr.release()


def test_08_existing_pipeline_receives_frames():
    """8. Existing pipeline still receives frames from CameraManager."""
    mgr = CameraManager(source="pc")
    mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    mock_cam = MagicMock()
    mock_cam.get_frame.return_value = (True, mock_frame)
    mock_cam.is_connected = True
    mgr.camera = mock_cam

    success, frame = mgr.get_frame()
    assert success is True
    assert frame is not None
    assert frame.shape == (720, 1280, 3)

    # Test read_frame alias
    success, frame = mgr.read_frame()
    assert success is True
    assert frame is not None
    mgr.release()


def test_09_hod_dashboard_has_camera_controls(client):
    """9. HOD Live Monitoring page contains all 4 camera controls, stop button, and status indicators."""
    response = client.get("/hod/dashboard")
    assert response.status_code == 200
    text = response.text
    assert "CAMERA SOURCE" in text
    assert "PC CAMERA" in text
    assert "DROIDCAM" in text
    assert "EXTENSION CAMERA" in text
    assert "ESP32 WI-FI CAMERA" in text
    assert "btn-stop-camera" in text
    assert "btn-start-camera" in text
    assert "active-cam-status-pill" in text
    assert "active-cam-res" in text
    assert "active-cam-fps" in text
    assert "10.140.159.218" in text
    assert "4747" in text


def test_10_advisor_dashboard_does_not_have_camera_switch_controls(client):
    """10. Advisor dashboard only monitors assigned stream and does not expose camera configuration controls."""
    response = client.get("/advisor/dashboard")
    assert response.status_code == 200
    text = response.text
    assert "card-cam-pc" not in text
    assert "card-cam-droidcam" not in text
    assert "card-cam-extension" not in text
    assert "card-cam-esp32" not in text
    assert "btn-stop-camera" not in text


def test_11_camera_stop_and_status_api(client):
    """11. Camera STOP releases camera worker and updates status to DISCONNECTED."""
    stop_resp = client.post("/api/camera/stop")
    assert stop_resp.status_code == 200
    stop_data = stop_resp.json()
    assert stop_data["success"] is True
    assert stop_data["data"]["camera_online"] is False

    status_resp = client.get("/api/camera/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["success"] is True
    assert status_data["data"]["camera_online"] is False


def test_12_switching_releases_previous_camera():
    """12. Switching camera source strictly releases prior camera to prevent simultaneous connections."""
    with patch.object(SmartBoardCamera, "_initialize_camera", return_value=True):
        mgr = CameraManager(source="pc")
        initial_cam = mgr.camera
        initial_release_spy = MagicMock()
        initial_cam.release = initial_release_spy

        def mock_init_droidcam(self_dc):
            self_dc.is_connected = True
            return True

        with patch.object(DroidCamCamera, "_initialize_camera", mock_init_droidcam):
            with patch.object(DroidCamCamera, "get_frame", return_value=(True, np.zeros((720, 1280, 3), dtype=np.uint8))):
                success, msg = mgr.switch_source("droidcam", host="10.140.159.218", port=4747)
                assert mgr.current_source == "droidcam"
                # Verify the prior PC camera release() was called
                assert initial_release_spy.called
                assert isinstance(mgr.camera, DroidCamCamera)

        droid_cam = mgr.camera
        droid_release_spy = MagicMock()
        droid_cam.release = droid_release_spy

        # Switch to Extension
        with patch.object(SmartBoardCamera, "_initialize_camera", return_value=True):
            success, msg = mgr.switch_source("extension", index=1)
            assert mgr.current_source == "extension"
            # Verify prior DroidCam release() was called
            assert droid_release_spy.called
            assert isinstance(mgr.camera, SmartBoardCamera)

        mgr.release()


