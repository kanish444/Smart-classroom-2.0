import cv2
import os
import time
import threading
import numpy as np
from typing import Tuple, Any, Optional, Dict
from loguru import logger
from .base_camera import BaseCamera, BooleanCallable


class SmartBoardCamera(BaseCamera):
    """
    High-performance, low-latency camera implementation for the Smart Board / PC webcam.
    Features:
    - Dedicated background capture thread to prevent OpenCV DirectShow buffer accumulation
    - Bounded single-frame buffer (maxsize=1) with stale frame dropping
    - Zero-latency latest-frame retrieval
    - Explicit resource lifecycle management (connect/release without hardware leaks)
    - Full status tracking: DISCONNECTED, CONNECTING, CONNECTED, STREAMING, ERROR, STOPPED
    - Actual vs requested resolution and FPS verification
    - Thread-safe acquisition and clean release
    """

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.camera_index = int(camera_index)
        self._requested_width = int(width)
        self._requested_height = int(height)
        self._requested_fps = int(fps)

        self._actual_width = 0
        self._actual_height = 0
        self._reported_fps = 0.0
        self._resolution_accepted = False
        self._fps_accepted = False

        # Status tracking: DISCONNECTED, CONNECTING, CONNECTED, STREAMING, ERROR, STOPPED
        self._status = "DISCONNECTED"

        # True FPS calculation variables
        self._frame_count = 0
        self._start_time = 0.0

        self.cap: Optional[cv2.VideoCapture] = None
        self._is_connected = False
        self.last_error_message = ""

        # Threaded acquisition lock & low-latency single-frame buffer (maxsize=1)
        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._stop_reader = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None

        self._initialize_camera()

    @property
    def is_connected(self) -> BooleanCallable:
        """Returns connection state as BooleanCallable for both property and method access."""
        return BooleanCallable(1 if self._is_connected else 0)

    @is_connected.setter
    def is_connected(self, value: bool):
        self._is_connected = bool(value)

    def _cleanup_prior_resources(self):
        """Safely terminates existing capture thread and releases OpenCV handle."""
        self._stop_reader.set()
        if self._reader_thread and self._reader_thread.is_alive():
            try:
                self._reader_thread.join(timeout=0.6)
            except Exception:
                pass
        self._reader_thread = None

        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"SmartBoardCamera: Exception releasing prior VideoCapture: {e}")
            self.cap = None

        with self._frame_lock:
            self._latest_frame = None

    def _initialize_camera(self) -> bool:
        """Opens physical webcam device, sets parameters, verifies acceptance, grabs test frame, and spawns reader thread."""
        with self._lock:
            self._cleanup_prior_resources()
            self._status = "CONNECTING"

            logger.info(f"SmartBoardCamera: Initializing camera index {self.camera_index}...")

            # Try standard OpenCV backend first, with DirectShow on Windows
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            try:
                self.cap = cv2.VideoCapture(self.camera_index, backend)
            except Exception as e:
                logger.warning(f"SmartBoardCamera: Error opening camera with backend {backend}: {e}")
                self.cap = None

            if self.cap is None or not self.cap.isOpened():
                logger.warning(f"SmartBoardCamera: DirectShow open failed. Falling back to default backend...")
                try:
                    self.cap = cv2.VideoCapture(self.camera_index)
                except Exception as e:
                    self.last_error_message = f"Failed to instantiate VideoCapture for index {self.camera_index}: {e}"
                    logger.error(self.last_error_message)
                    self._is_connected = False
                    self._status = "ERROR"
                    return False

            if not self.cap.isOpened():
                self.last_error_message = f"Could not open camera device at index {self.camera_index}."
                logger.error(self.last_error_message)
                self._is_connected = False
                self._status = "ERROR"
                return False

            # Request buffer size 1 to prevent driver-level buffer bloat
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # Configure requested resolution and FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
            try:
                self.cap.set(cv2.CAP_PROP_FPS, self._requested_fps)
            except Exception:
                pass

            # Read back properties reported by driver
            prop_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self._requested_width)
            prop_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self._requested_height)
            self._reported_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or self._requested_fps)

            # Test initial frame grab with settling retries
            ret, frame = False, None
            for _ in range(5):
                ret, frame = self.cap.read()
                if ret and self._validate_frame(frame):
                    break
                time.sleep(0.05)

            if not ret or not self._validate_frame(frame):
                self.last_error_message = f"Camera index {self.camera_index} opened but failed to read initial frame."
                logger.warning(self.last_error_message)
                self._is_connected = self.cap.isOpened()
                if not self._is_connected:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    self._status = "ERROR"
                    return False
                self._status = "CONNECTED"
            else:
                self._actual_height, self._actual_width = frame.shape[:2]
                self._is_connected = True
                self._status = "CONNECTED"
                self.last_error_message = ""
                with self._frame_lock:
                    self._latest_frame = frame

            # Verify whether requested resolution and FPS were accepted by hardware
            self._resolution_accepted = (
                self._actual_width == self._requested_width and self._actual_height == self._requested_height
            )
            self._fps_accepted = abs(self._reported_fps - self._requested_fps) < 1.0

            logger.info(
                f"SmartBoardCamera index {self.camera_index} connected: "
                f"Actual: {self._actual_width}x{self._actual_height} @ ~{self._reported_fps:.1f} FPS "
                f"(Requested: {self._requested_width}x{self._requested_height} @ {self._requested_fps} FPS, "
                f"Resolution accepted: {self._resolution_accepted}, FPS accepted: {self._fps_accepted})"
            )

            self._frame_count = 1
            self._start_time = time.time()

            # Start low-latency background capture worker thread
            self._stop_reader.clear()
            self._reader_thread = threading.Thread(
                target=self._capture_worker,
                name=f"SmartBoardCameraWorker_{self.camera_index}",
                daemon=True
            )
            self._reader_thread.start()
            self._status = "STREAMING"
            return True

    def _capture_worker(self):
        """
        Continuously drains frames from VideoCapture to prevent OpenCV DirectShow buffer accumulation.
        Maintains ONLY the latest frame in a single-slot buffer (size=1).
        Discards stale frames when AI processing is busy.
        """
        self._status = "STREAMING"
        while not self._stop_reader.is_set():
            with self._lock:
                if not self.cap or not self.cap.isOpened():
                    break
                try:
                    ret, frame = self.cap.read()
                except Exception as e:
                    logger.error(f"SmartBoardCamera worker exception: {e}")
                    ret, frame = False, None

            if not ret or not self._validate_frame(frame):
                time.sleep(0.005)
                continue

            with self._frame_lock:
                self._latest_frame = frame
                self._frame_count += 1

    def _validate_frame(self, frame: Any) -> bool:
        """Basic generic frame validation."""
        if frame is None:
            return False
        if not isinstance(frame, np.ndarray):
            return False
        if frame.size == 0:
            return False
        return True

    def get_frame(self) -> Tuple[bool, Any]:
        """
        Returns the newest video frame with lowest possible latency (from bounded single-frame buffer).
        """
        if not self._is_connected:
            return False, None

        # Prefer newest frame from dedicated capture worker (bounded buffer maxsize=1)
        if self._reader_thread and self._reader_thread.is_alive():
            with self._frame_lock:
                if self._latest_frame is not None:
                    return True, self._latest_frame
            time.sleep(0.005)
            with self._frame_lock:
                if self._latest_frame is not None:
                    return True, self._latest_frame

        # Fallback direct read (used in test mocks or if thread is inactive)
        if self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
            except Exception as e:
                logger.error(f"SmartBoardCamera direct read exception: {e}")
                ret, frame = False, None

            if ret and self._validate_frame(frame):
                self._frame_count += 1
                return True, frame

        return False, None

    def read_frame(self) -> Tuple[bool, Any]:
        """Alias for get_frame to satisfy frame interface."""
        return self.get_frame()

    def get_true_fps(self) -> float:
        """Calculate the actual measured FPS based on successfully read frames."""
        if self._frame_count <= 1:
            return 0.0
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return round(self._frame_count / elapsed, 2)

    def get_fps(self) -> float:
        """Returns measured FPS if available, else reported FPS."""
        measured = self.get_true_fps()
        if measured > 0:
            return measured
        return round(self._reported_fps, 1)

    def get_resolution(self) -> Tuple[int, int]:
        """Returns actual camera resolution (width, height)."""
        return (self._actual_width, self._actual_height)

    def connect(self) -> bool:
        """Explicit connect method to check or re-establish connection."""
        if self._is_connected and self.cap and self.cap.isOpened():
            return True
        return self._initialize_camera()

    def disconnect(self) -> None:
        """Alias for release to cleanly shut down camera."""
        self.release()

    def release(self) -> None:
        """Stops capture worker thread, releases VideoCapture, and resets state to STOPPED."""
        self._stop_reader.set()
        if self._reader_thread and self._reader_thread.is_alive():
            try:
                self._reader_thread.join(timeout=1.0)
            except Exception:
                pass
        self._reader_thread = None

        with self._lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception as e:
                    logger.warning(f"SmartBoardCamera: Error releasing VideoCapture: {e}")
                self.cap = None

            with self._frame_lock:
                self._latest_frame = None

            self._is_connected = False
            self._status = "STOPPED"
            logger.info(f"SmartBoardCamera index {self.camera_index} released cleanly.")

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

    def get_status(self) -> Dict[str, Any]:
        """Returns full diagnostic and runtime status dictionary."""
        is_conn = bool(self.is_connected)
        fps_val = self.get_fps()
        return {
            "status": self._status,
            "camera_index": self.camera_index,
            "connected": is_conn,
            "resolution": f"{self._actual_width}x{self._actual_height}",
            "actual_resolution": f"{self._actual_width}x{self._actual_height}",
            "requested_resolution": f"{self._requested_width}x{self._requested_height}",
            "resolution_accepted": self._resolution_accepted,
            "fps": fps_val,
            "reported_fps": round(self._reported_fps, 1),
            "measured_fps": self.get_true_fps(),
            "fps_accepted": self._fps_accepted,
            "frames_read": self._frame_count,
            "last_error": self.last_error_message
        }
