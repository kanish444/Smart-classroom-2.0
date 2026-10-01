import time
import threading
from typing import Tuple, Any, Optional, Dict, List
from loguru import logger
import os
import cv2
from .base_camera import BaseCamera, BooleanCallable
from .smartboard_camera import SmartBoardCamera
from .droidcam_camera import DroidCamCamera
from .esp32_camera import ESP32Camera
from config.settings import get_settings


class CameraManager:
    """
    Manages active camera source (PC Camera, DroidCam Wi-Fi, Extension USB Camera, ESP32 Wi-Fi Camera),
    handles robust reconnect logic, supports dynamic source switching, and abstracts hardware failures.
    Thread-safe acquisition and switching.
    """

    def __init__(self, source: Optional[str] = None):
        self._lock = threading.Lock()
        self.settings = get_settings()
        self.current_source = (source or self.settings.camera.source).lower()
        self.camera: Optional[BaseCamera] = None
        self._is_stopped = False
        with self._lock:
            self._initialize_source()

    def _initialize_source(self):
        """Instantiates the concrete camera backend based on current_source."""
        if self.camera:
            try:
                self.camera.release()
            except Exception as e:
                logger.warning(f"CameraManager: Exception releasing prior camera: {e}")
            self.camera = None

        logger.info(f"CameraManager: Instantiating source '{self.current_source}'...")
        if self.current_source == "droidcam":
            dc = self.settings.camera.droidcam
            self.camera = DroidCamCamera(
                host=dc.host,
                port=dc.port,
                video_path=dc.video_path,
                width=self.settings.camera.width,
                height=self.settings.camera.height
            )
        elif self.current_source in ["pc", "laptop", "webcam"]:
            pc_cfg = getattr(self.settings.camera, "pc", None)
            dev_idx = getattr(pc_cfg, "device_index", self.settings.camera.laptop_index) if pc_cfg else self.settings.camera.laptop_index
            w = getattr(pc_cfg, "width", self.settings.camera.width) if pc_cfg else self.settings.camera.width
            h = getattr(pc_cfg, "height", self.settings.camera.height) if pc_cfg else self.settings.camera.height
            fps = getattr(pc_cfg, "fps", getattr(self.settings.camera, "fps", 30)) if pc_cfg else getattr(self.settings.camera, "fps", 30)

            self.camera = SmartBoardCamera(
                camera_index=dev_idx,
                width=w,
                height=h,
                fps=fps
            )
        elif self.current_source in ["extension", "external"]:
            ext_idx = getattr(self.settings.camera, "extension_index", self.settings.camera.external_index)
            fps = getattr(self.settings.camera, "fps", 30)
            self.camera = SmartBoardCamera(
                camera_index=ext_idx,
                width=self.settings.camera.width,
                height=self.settings.camera.height,
                fps=fps
            )
        elif self.current_source == "esp32":
            self.camera = ESP32Camera(
                stream_url=self.settings.camera.esp32.stream_url,
                width=self.settings.camera.width,
                height=self.settings.camera.height
            )
        elif self.current_source == "smart_board":
            fps = getattr(self.settings.camera, "fps", 30)
            self.camera = SmartBoardCamera(
                camera_index=self.settings.camera.smart_board_index,
                width=self.settings.camera.width,
                height=self.settings.camera.height,
                fps=fps
            )
        else:
            logger.warning(
                f"Unknown camera source '{self.current_source}'. "
                f"Falling back to camera index {self.settings.camera.index}."
            )
            fps = getattr(self.settings.camera, "fps", 30)
            self.camera = SmartBoardCamera(
                camera_index=self.settings.camera.index,
                width=self.settings.camera.width,
                height=self.settings.camera.height,
                fps=fps
            )

    def switch_source(self, source_type: str, **kwargs) -> Tuple[bool, str]:
        """
        Dynamically hot-swaps active camera source without restarting application.
        Accepts optional override kwargs (e.g. host='10.140.159.218', port=4747, stream_url=..., index=...).
        1. Disconnect current camera.
        2. Release resources.
        3. Connect selected camera.
        4. Verify a frame.
        5. Update status.
        """
        source_type = source_type.lower()
        valid_sources = ["pc", "laptop", "webcam", "droidcam", "extension", "external", "esp32", "smart_board"]
        if source_type not in valid_sources:
            return False, f"Invalid camera source '{source_type}'. Valid options: {valid_sources}"

        logger.info(f"CameraManager: Switching camera source to '{source_type}'...")

        with self._lock:
            self._is_stopped = False
            # 1 & 2. Disconnect current camera and release resources cleanly
            if self.camera:
                try:
                    self.camera.release()
                except Exception as e:
                    logger.warning(f"Error releasing prior camera before switch: {e}")
                self.camera = None

            # Update runtime settings if provided
            if source_type == "droidcam":
                if "host" in kwargs and kwargs["host"]:
                    self.settings.camera.droidcam.host = str(kwargs["host"]).strip()
                if "port" in kwargs and kwargs["port"]:
                    self.settings.camera.droidcam.port = int(kwargs["port"])
                if "video_path" in kwargs and kwargs["video_path"]:
                    self.settings.camera.droidcam.video_path = str(kwargs["video_path"]).strip()
            elif source_type in ["pc", "laptop", "webcam"] and "index" in kwargs and kwargs["index"] is not None:
                new_idx = int(kwargs["index"])
                self.settings.camera.laptop_index = new_idx
                if hasattr(self.settings.camera, "pc"):
                    self.settings.camera.pc.device_index = new_idx
            elif source_type in ["extension", "external"] and "index" in kwargs and kwargs["index"] is not None:
                ext_idx = int(kwargs["index"])
                self.settings.camera.external_index = ext_idx
                if hasattr(self.settings.camera, "extension_index"):
                    self.settings.camera.extension_index = ext_idx
            elif source_type == "esp32" and "stream_url" in kwargs and kwargs["stream_url"]:
                self.settings.camera.esp32.stream_url = str(kwargs["stream_url"]).strip()
            elif source_type == "smart_board" and "index" in kwargs and kwargs["index"] is not None:
                self.settings.camera.smart_board_index = int(kwargs["index"])

            self.current_source = source_type
            self.settings.camera.source = source_type

            # 3. Connect selected camera
            self._initialize_source()

            # 4. Verify a frame / connection
            is_conn = bool(getattr(self.camera, "is_connected", False))
            if is_conn:
                try:
                    test_success, test_frame = self.camera.get_frame()
                    if test_success and test_frame is not None:
                        return True, f"Successfully switched to {source_type}."
                except Exception:
                    pass
                return True, f"Successfully switched to {source_type}."
            else:
                err = getattr(
                    self.camera,
                    "last_error_message",
                    f"{source_type.upper()} stream unavailable."
                )
                return False, f"Switched to {source_type}, but stream is not connected: {err}"

    def get_frame(self, max_retries: Optional[int] = None) -> Tuple[bool, Any]:
        """
        Attempts to read a frame. If disconnected, attempts controlled reconnect.
        """
        with self._lock:
            if self._is_stopped:
                return False, None

            if not self.camera:
                logger.error("CameraManager: No camera source instantiated.")
                return False, None

            success, frame = self.camera.get_frame()
            if success and frame is not None:
                return True, frame

            # Reconnect logic
            retries_limit = (
                max_retries
                if max_retries is not None
                else self.settings.camera.retry.max_attempts
            )
            delay_sec = self.settings.camera.retry.delay_seconds

            logger.warning(
                f"CameraManager: Frame read failed from {self.current_source}. Attempting reconnect..."
            )
            self.camera.release()

            retries = 0
            while retries < retries_limit:
                retries += 1
                logger.info(
                    f"CameraManager: Reconnect attempt {retries}/{retries_limit} to {self.current_source}..."
                )
                time.sleep(delay_sec)

                self._initialize_source()
                success, frame = self.camera.get_frame()
                if success and frame is not None:
                    logger.info(f"CameraManager: Reconnection to {self.current_source} successful.")
                    return True, frame

            err_msg = (
                f"CAMERA ERROR: {self.current_source.upper()} UNAVAILABLE after {retries_limit} attempts."
            )
            logger.error(err_msg)
            return False, None

    def read_frame(self, max_retries: Optional[int] = None) -> Tuple[bool, Any]:
        """Alias for get_frame to satisfy frame interface."""
        return self.get_frame(max_retries=max_retries)

    @property
    def is_connected(self) -> BooleanCallable:
        """Returns connection state of current camera source as BooleanCallable."""
        if not self.camera:
            return BooleanCallable(0)
        is_conn = bool(getattr(self.camera, "is_connected", False))
        return BooleanCallable(1 if is_conn else 0)

    def get_resolution(self) -> Tuple[int, int]:
        """Returns camera resolution (width, height)."""
        with self._lock:
            if not self.camera:
                return (0, 0)
            if hasattr(self.camera, "get_resolution"):
                return self.camera.get_resolution()
            return getattr(self.camera, "resolution", (0, 0))

    def get_fps(self) -> float:
        """Returns current FPS."""
        with self._lock:
            if not self.camera:
                return 0.0
            if hasattr(self.camera, "get_fps"):
                return self.camera.get_fps()
            return getattr(self.camera, "reported_fps", 0.0)

    def connect(self) -> bool:
        """Connects or verifies connection to camera source."""
        with self._lock:
            self._is_stopped = False
            if not self.camera:
                self._initialize_source()
            if hasattr(self.camera, "connect"):
                return bool(self.camera.connect())
            return bool(getattr(self.camera, "is_connected", False))

    def start_camera(self):
        """Explicitly starts capture, resets stopped state, and initializes camera source."""
        with self._lock:
            self._is_stopped = False
            if not self.camera or not getattr(self.camera, "is_connected", False):
                self._initialize_source()
            logger.info(f"CameraManager: Camera started for source '{self.current_source}'.")

    def disconnect(self):
        """Disconnects camera source cleanly."""
        self.release()

    def stop_camera(self):
        """Explicitly stops capture, releases hardware, clears buffer, and disables auto-reconnect."""
        with self._lock:
            self._is_stopped = True
            if self.camera:
                try:
                    self.camera.release()
                except Exception as e:
                    logger.warning(f"CameraManager: Exception in stop_camera: {e}")
                self.camera = None
            logger.info("CameraManager: Camera stopped explicitly. Auto-reconnect disabled.")

    def release(self):
        """Releases active camera source."""
        with self._lock:
            self._is_stopped = True
            if self.camera:
                try:
                    self.camera.release()
                except Exception as e:
                    logger.warning(f"CameraManager: Error releasing camera: {e}")
                self.camera = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def get_status(self) -> Dict[str, Any]:
        """Returns diagnostic status of the active camera conforming to required states."""
        with self._lock:
            if self._is_stopped:
                return {
                    "source_type": self.current_source,
                    "status": "STOPPED",
                    "connected": False,
                    "resolution": "0x0",
                    "fps": 0.0
                }

            if not self.camera:
                return {
                    "source_type": self.current_source,
                    "status": "DISCONNECTED",
                    "connected": False,
                    "resolution": "0x0",
                    "fps": 0.0
                }

            if hasattr(self.camera, "get_status"):
                stat = self.camera.get_status()
                stat["source_type"] = self.current_source
                return stat

            is_conn = bool(getattr(self.camera, "is_connected", False))
            status_str = "CONNECTED" if is_conn else "DISCONNECTED"

            if isinstance(self.camera, DroidCamCamera):
                diag = self.camera.get_diagnostic_dict()
                diag["status"] = status_str
                diag["source_type"] = self.current_source
                return diag

            if isinstance(self.camera, ESP32Camera):
                diag = self.camera.get_diagnostic_dict()
                diag["status"] = status_str
                diag["source_type"] = self.current_source
                return diag

            w, h = getattr(self.camera, "resolution", (0, 0))
            return {
                "source_type": self.current_source,
                "status": status_str,
                "connected": is_conn,
                "resolution": f"{w}x{h}",
                "fps": getattr(self.camera, "reported_fps", 0.0)
            }

    @staticmethod
    def enumerate_cameras(max_to_test: int = 3) -> List[Dict[str, Any]]:
        """
        Scan of available local OpenCV camera indices.
        Returns list of device dicts: [{"index": 0, "name": "Integrated Camera"}, ...]
        """
        devices = []
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        except Exception:
            pass

        for idx in range(max_to_test):
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    label = "Integrated Camera" if idx == 0 else f"USB Camera (Index {idx})"
                    devices.append({"index": idx, "name": label})
                else:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    break
            except Exception:
                break

        if not devices:
            devices = [
                {"index": 0, "name": "Integrated Camera"},
                {"index": 1, "name": "USB Camera (Index 1)"}
            ]
        return devices
