import time
import threading
from typing import Tuple, Any, Optional, Dict
from loguru import logger
from .base_camera import BaseCamera
from .smartboard_camera import SmartBoardCamera
from .droidcam_camera import DroidCamCamera
from config.settings import get_settings


class CameraManager:
    """
    Manages active camera source (DroidCam Wi-Fi, Laptop Camera, Smart Board Camera, External Camera),
    handles robust reconnect logic, supports dynamic source switching, and abstracts hardware failures.
    Thread-safe acquisition and switching.
    """

    def __init__(self, source: Optional[str] = None):
        self._lock = threading.Lock()
        self.settings = get_settings()
        self.current_source = (source or self.settings.camera.source).lower()
        self.camera: Optional[BaseCamera] = None
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
        elif self.current_source == "laptop":
            self.camera = SmartBoardCamera(
                camera_index=self.settings.camera.laptop_index,
                width=self.settings.camera.width,
                height=self.settings.camera.height
            )
        elif self.current_source == "smart_board":
            self.camera = SmartBoardCamera(
                camera_index=self.settings.camera.smart_board_index,
                width=self.settings.camera.width,
                height=self.settings.camera.height
            )
        elif self.current_source == "external":
            self.camera = SmartBoardCamera(
                camera_index=self.settings.camera.external_index,
                width=self.settings.camera.width,
                height=self.settings.camera.height
            )
        else:
            logger.warning(
                f"Unknown camera source '{self.current_source}'. "
                f"Falling back to camera index {self.settings.camera.index}."
            )
            self.camera = SmartBoardCamera(
                camera_index=self.settings.camera.index,
                width=self.settings.camera.width,
                height=self.settings.camera.height
            )

    def switch_source(self, source_type: str, **kwargs) -> Tuple[bool, str]:
        """
        Dynamically hot-swaps active camera source without restarting application.
        Accepts optional override kwargs (e.g. host='10.140.159.218', port=4747).
        """
        source_type = source_type.lower()
        valid_sources = ["droidcam", "laptop", "smart_board", "external"]
        if source_type not in valid_sources:
            return False, f"Invalid camera source '{source_type}'. Valid options: {valid_sources}"

        logger.info(f"CameraManager: Switching camera source to '{source_type}'...")

        with self._lock:
            # Update runtime settings if provided
            if source_type == "droidcam":
                if "host" in kwargs and kwargs["host"]:
                    self.settings.camera.droidcam.host = str(kwargs["host"]).strip()
                if "port" in kwargs and kwargs["port"]:
                    self.settings.camera.droidcam.port = int(kwargs["port"])
                if "video_path" in kwargs and kwargs["video_path"]:
                    self.settings.camera.droidcam.video_path = str(kwargs["video_path"]).strip()
            elif source_type == "laptop" and "index" in kwargs:
                self.settings.camera.laptop_index = int(kwargs["index"])
            elif source_type == "smart_board" and "index" in kwargs:
                self.settings.camera.smart_board_index = int(kwargs["index"])
            elif source_type == "external" and "index" in kwargs:
                self.settings.camera.external_index = int(kwargs["index"])

            self.current_source = source_type
            self.settings.camera.source = source_type
            self._initialize_source()

            is_conn = getattr(self.camera, "is_connected", False)
            if is_conn:
                return True, f"Successfully switched to {source_type}."
            else:
                err = getattr(
                    self.camera,
                    "last_error_message",
                    f"{source_type.capitalize()} stream unavailable."
                )
                return False, f"Switched to {source_type}, but stream is not connected: {err}"

    def get_frame(self, max_retries: Optional[int] = None) -> Tuple[bool, Any]:
        """
        Attempts to read a frame. If disconnected, attempts controlled reconnect.
        """
        with self._lock:
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

    def connect(self) -> bool:
        """Connects or verifies connection to camera source."""
        with self._lock:
            if not self.camera:
                self._initialize_source()
            if hasattr(self.camera, "connect"):
                return self.camera.connect()
            return getattr(self.camera, "is_connected", False)

    def disconnect(self):
        """Disconnects camera source."""
        self.release()

    def release(self):
        """Releases active camera source."""
        with self._lock:
            if self.camera:
                self.camera.release()

    def get_status(self) -> dict:
        """Returns diagnostic status of the active camera."""
        with self._lock:
            if not self.camera:
                return {
                    "source_type": self.current_source,
                    "status": "NOT_INITIALIZED",
                    "connected": False
                }

            is_conn = getattr(self.camera, "is_connected", False)
            status_str = "CONNECTED" if is_conn else "DISCONNECTED"

            if isinstance(self.camera, DroidCamCamera):
                diag = self.camera.get_diagnostic_dict()
                diag["status"] = status_str
                return diag

            if isinstance(self.camera, SmartBoardCamera):
                return {
                    "source_type": self.current_source,
                    "camera_index": self.camera.camera_index,
                    "status": status_str,
                    "connected": is_conn,
                    "resolution": f"{self.camera._actual_width}x{self.camera._actual_height}",
                    "reported_fps": self.camera.reported_fps,
                    "measured_fps": self.camera.get_true_fps(),
                    "frames_read": self.camera._frame_count
                }

            return {
                "source_type": self.current_source,
                "status": status_str,
                "connected": is_conn
            }

