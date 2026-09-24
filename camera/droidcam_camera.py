import time
import socket
import urllib.request
import threading
import cv2
import numpy as np
from typing import Tuple, Any, Optional, Dict
from loguru import logger
from .base_camera import BaseCamera


class DroidCamCamera(BaseCamera):
    """
    Camera implementation for DroidCam over Wi-Fi / IP stream.
    Connects to the video stream endpoint (e.g. http://<IP>:4747/video).
    Includes socket health probing, OpenCV capture, and graceful error diagnostics.
    """

    def __init__(
        self,
        host: str = "10.140.159.218",
        port: int = 4747,
        video_path: str = "/video",
        width: int = 1280,
        height: int = 720
    ):
        self.host = host
        self.port = int(port)
        self.video_path = video_path if video_path.startswith("/") else f"/{video_path}"
        self._requested_width = width
        self._requested_height = height

        self.base_url = f"http://{self.host}:{self.port}"
        self.video_url = f"{self.base_url}{self.video_path}"

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
    def probe_endpoint(host: str, port: int, timeout: float = 1.5) -> Tuple[bool, str]:
        """
        Fast TCP connection check to verify if the DroidCam remote port is listening.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            s.close()
            return True, "Port reachable"
        except socket.timeout:
            return False, "Connection timed out. Phone and laptop may not be on the same Wi-Fi network."
        except ConnectionRefusedError:
            return False, "Connection refused. DroidCam app may not be running on the phone."
        except Exception as e:
            return False, f"Network probe failed: {str(e)}"

    def _initialize_camera(self) -> bool:
        """Initializes OpenCV video capture from the DroidCam /video stream."""
        logger.info(f"DroidCamCamera: Probing {self.host}:{self.port}...")
        reachable, reason = self.probe_endpoint(self.host, self.port, timeout=1.5)
        if not reachable:
            self.last_error_message = (
                f"DroidCam video stream unavailable at {self.video_url}.\n"
                f"Diagnostic reason: {reason}\n"
                "Possible causes:\n"
                "1. Phone and laptop not connected to the same Wi-Fi network\n"
                "2. DroidCam mobile app is not currently open/streaming\n"
                f"3. Incorrect IP ({self.host}) or port ({self.port})\n"
                "4. Windows Firewall is blocking incoming/outgoing connections on port 4747"
            )
            logger.warning(self.last_error_message)
            self.is_connected = False
            return False

        logger.info(f"DroidCamCamera: Opening video endpoint {self.video_url} via OpenCV...")
        try:
            self.cap = cv2.VideoCapture(self.video_url)
        except Exception as e:
            self.last_error_message = f"Failed to instantiate VideoCapture for {self.video_url}: {e}"
            logger.error(self.last_error_message)
            self.is_connected = False
            return False

        if not self.cap.isOpened():
            self.last_error_message = (
                f"DroidCam video stream unavailable. OpenCV could not open {self.video_url}.\n"
                "Verify that DroidCam is running and video streaming is enabled in the app."
            )
            logger.error(self.last_error_message)
            self.is_connected = False
            return False

        # Attempt to read first test frame
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            self.last_error_message = f"Connected to {self.video_url} but failed to decode initial video frame."
            logger.error(self.last_error_message)
            self.is_connected = False
            self.cap.release()
            return False

        self._actual_height, self._actual_width = frame.shape[:2]
        self._reported_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.is_connected = True
        self.last_error_message = ""
        self._frame_count = 1
        self._start_time = time.time()

        logger.info(
            f"DroidCamCamera connected successfully! "
            f"Resolution: {self._actual_width}x{self._actual_height} @ ~{self._reported_fps:.1f} reported FPS"
        )
        return True

    def get_frame(self) -> Tuple[bool, Any]:
        """Reads the next video frame from the DroidCam stream."""
        if not self.is_connected or not self.cap or not self.cap.isOpened():
            return False, None

        try:
            ret, frame = self.cap.read()
        except Exception as e:
            logger.error(f"DroidCam exception while reading frame: {e}")
            ret, frame = False, None

        if not ret or frame is None or frame.size == 0:
            logger.warning("DroidCam failed to return a valid frame. Stream may be disconnected.")
            self.is_connected = False
            self.last_error_message = "Stream interrupted or disconnected."
            return False, None

        self._frame_count += 1
        return True, frame

    def get_true_fps(self) -> float:
        """Calculates actual measured FPS based on successfully decoded frames."""
        if self._frame_count == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return round(self._frame_count / elapsed, 2)

    def connect(self) -> bool:
        """Connects or verifies connection to DroidCam."""
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
                    logger.warning(f"Error releasing DroidCam capture: {e}")
                self.cap = None
            self.is_connected = False
            logger.info("DroidCamCamera released.")

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._actual_width, self._actual_height)

    @property
    def reported_fps(self) -> float:
        return self._reported_fps

    def get_diagnostic_dict(self) -> Dict[str, Any]:
        """Returns structured diagnostic dictionary for health and API status."""
        return {
            "source_type": "droidcam",
            "host": self.host,
            "port": self.port,
            "base_url": self.base_url,
            "video_url": self.video_url,
            "connected": self.is_connected,
            "resolution": f"{self._actual_width}x{self._actual_height}",
            "reported_fps": self._reported_fps,
            "measured_fps": self.get_true_fps(),
            "frames_read": self._frame_count,
            "last_error": self.last_error_message
        }
