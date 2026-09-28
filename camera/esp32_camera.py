import time
import socket
import urllib.parse
import threading
import cv2
import numpy as np
from typing import Tuple, Any, Optional, Dict
from loguru import logger
from .base_camera import BaseCamera


class ESP32Camera(BaseCamera):
    """
    Camera implementation for ESP32 Wi-Fi Camera.
    Connects to the video stream endpoint (e.g. http://192.168.1.100:81/stream).
    Includes socket health probing, OpenCV capture, and graceful error diagnostics.
    """

    def __init__(
        self,
        stream_url: str = "http://192.168.1.100:81/stream",
        width: int = 1280,
        height: int = 720
    ):
        self.stream_url = stream_url.strip()
        self._requested_width = width
        self._requested_height = height

        self._actual_width = 0
        self._actual_height = 0
        self._reported_fps = 0.0

        self._frame_count = 0
        self._start_time = 0.0
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.last_error_message = ""

        # Threaded acquisition lock
        self._lock = threading.Lock()

        self._initialize_camera()

    @staticmethod
    def probe_endpoint(stream_url: str, timeout: float = 1.5) -> Tuple[bool, str]:
        """
        Fast TCP connection check to verify if the ESP32 remote host and port are listening.
        """
        try:
            parsed = urllib.parse.urlparse(stream_url)
            host = parsed.hostname or stream_url
            port = parsed.port
            if not port:
                port = 443 if parsed.scheme == "https" else 80
        except Exception as e:
            return False, f"Invalid stream URL '{stream_url}': {e}"

        if not host:
            return False, "Could not determine host from stream URL."

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            s.close()
            return True, f"Port {port} on {host} reachable"
        except socket.timeout:
            return False, f"Connection timed out. Host {host}:{port} did not respond within {timeout}s."
        except ConnectionRefusedError:
            return False, f"Connection refused at {host}:{port}. ESP32 stream server may not be running."
        except Exception as e:
            return False, f"Network probe failed: {str(e)}"

    def _initialize_camera(self) -> bool:
        """Initializes OpenCV video capture from the ESP32 Wi-Fi video stream."""
        logger.info(f"ESP32Camera: Probing {self.stream_url}...")
        reachable, reason = self.probe_endpoint(self.stream_url, timeout=1.5)
        if not reachable:
            self.last_error_message = (
                f"ESP32 video stream unavailable at {self.stream_url}.\n"
                f"Diagnostic reason: {reason}\n"
                "Possible causes:\n"
                "1. ESP32 board and PC not connected to the same Wi-Fi network\n"
                "2. ESP32 camera web server is not running or powered off\n"
                f"3. Incorrect stream URL ({self.stream_url})\n"
                "4. Local firewall blocking connection"
            )
            logger.warning(self.last_error_message)
            self.is_connected = False
            return False

        logger.info(f"ESP32Camera: Opening video endpoint {self.stream_url} via OpenCV...")
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
        except Exception as e:
            self.last_error_message = f"Failed to instantiate VideoCapture for {self.stream_url}: {e}"
            logger.error(self.last_error_message)
            self.is_connected = False
            return False

        if not self.cap.isOpened():
            self.last_error_message = (
                f"ESP32 video stream unavailable. OpenCV could not open {self.stream_url}.\n"
                "Verify that ESP32 camera server is actively streaming MJPEG/video."
            )
            logger.error(self.last_error_message)
            self.is_connected = False
            return False

        # Attempt to read first test frame
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            self.last_error_message = f"Connected to {self.stream_url} but failed to decode initial video frame."
            logger.error(self.last_error_message)
            self.is_connected = False
            self.cap.release()
            return False

        self._actual_height, self._actual_width = frame.shape[:2]
        self._reported_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.is_connected = True
        self.last_error_message = ""
        self._frame_count = 1
        self._start_time = time.time()

        logger.info(
            f"ESP32Camera connected successfully! "
            f"Resolution: {self._actual_width}x{self._actual_height} @ ~{self._reported_fps:.1f} reported FPS"
        )
        return True

    def get_frame(self) -> Tuple[bool, Any]:
        """Reads the next video frame from the ESP32 stream."""
        if not self.is_connected or not self.cap or not self.cap.isOpened():
            return False, None

        try:
            ret, frame = self.cap.read()
        except Exception as e:
            logger.error(f"ESP32 exception while reading frame: {e}")
            ret, frame = False, None

        if not ret or frame is None or frame.size == 0:
            logger.warning("ESP32 failed to return a valid frame. Stream may be disconnected.")
            self.is_connected = False
            self.last_error_message = "Stream interrupted or disconnected."
            return False, None

        self._frame_count += 1
        return True, frame

    def read_frame(self) -> Tuple[bool, Any]:
        """Alias for get_frame to satisfy frame interface."""
        return self.get_frame()

    def get_true_fps(self) -> float:
        """Calculates actual measured FPS based on successfully decoded frames."""
        if self._frame_count == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return round(self._frame_count / elapsed, 2)

    def connect(self) -> bool:
        """Connects or verifies connection to ESP32 Camera."""
        if self.is_connected and self.cap and self.cap.isOpened():
            return True
        return self._initialize_camera()

    def disconnect(self) -> None:
        """Alias for release."""
        self.release()

    def release(self) -> None:
        """Releases the camera handle cleanly."""
        with self._lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception as e:
                    logger.warning(f"Error releasing ESP32 capture: {e}")
                self.cap = None
            self.is_connected = False
            logger.info("ESP32Camera released.")

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._actual_width, self._actual_height)

    @property
    def reported_fps(self) -> float:
        return self._reported_fps

    def get_diagnostic_dict(self) -> Dict[str, Any]:
        """Returns structured diagnostic dictionary for health and API status."""
        return {
            "source_type": "esp32",
            "stream_url": self.stream_url,
            "connected": self.is_connected,
            "resolution": f"{self._actual_width}x{self._actual_height}",
            "reported_fps": self._reported_fps,
            "measured_fps": self.get_true_fps(),
            "frames_read": self._frame_count,
            "last_error": self.last_error_message
        }
