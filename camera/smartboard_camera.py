import cv2
import os
import time
import threading
import numpy as np
from typing import Tuple, Any, Optional
from loguru import logger
from .base_camera import BaseCamera


class SmartBoardCamera(BaseCamera):
    """
    High-performance, low-latency camera implementation for the Smart Board / PC webcam.
    Features:
    - Dedicated background capture thread to prevent OpenCV DirectShow buffer accumulation
    - Bounded single-frame buffer (maxsize=1) with stale frame dropping
    - Zero-latency latest-frame retrieval
    - Explicit resource lifecycle management (connect/release without hardware leaks)
    - Thread-safe acquisition and clean release
    """

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.camera_index = int(camera_index)
        self._requested_width = width
        self._requested_height = height

        self._actual_width = 0
        self._actual_height = 0
        self._reported_fps = 0.0

        # True FPS calculation variables
        self._frame_count = 0
        self._start_time = 0.0

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.last_error_message = ""

        # Threaded acquisition lock & low-latency single-frame buffer (maxsize=1)
        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._stop_reader = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None

        self._initialize_camera()

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
        """Opens physical webcam device, sets parameters, grabs test frame, and spawns reader thread."""
        with self._lock:
            self._cleanup_prior_resources()

            logger.info(f"SmartBoardCamera: Initializing camera index {self.camera_index}...")

            # Try standard OpenCV backend first, with DirectShow on Windows
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            try:
                self.cap = cv2.VideoCapture(self.camera_index, backend)
            except Exception as e:
                logger.warning(f"SmartBoardCamera: Error opening camera with backend {backend}: {e}")
                self.cap = None

            if self.cap is None or not self.cap.isOpened():
                logger.warning(f"SmartBoardCamera: Failed with DirectShow. Falling back to default backend...")
                try:
                    self.cap = cv2.VideoCapture(self.camera_index)
                except Exception as e:
                    self.last_error_message = f"Failed to instantiate VideoCapture for index {self.camera_index}: {e}"
                    logger.error(self.last_error_message)
                    self.is_connected = False
                    return False

            if not self.cap.isOpened():
                self.last_error_message = f"Could not open camera device at index {self.camera_index}."
                logger.error(self.last_error_message)
                self.is_connected = False
                return False

            # Request buffer size 1 to prevent driver-level buffer bloat
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # Try to set requested resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)

            # Read back actual properties
            self._actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self._requested_width)
            self._actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self._requested_height)
            self._reported_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

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
                self.is_connected = self.cap.isOpened()
                if not self.is_connected:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    return False
            else:
                self._actual_height, self._actual_width = frame.shape[:2]
                self.is_connected = True
                self.last_error_message = ""
                with self._frame_lock:
                    self._latest_frame = frame

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

            logger.info(
                f"SmartBoardCamera connected with low-latency worker. "
                f"Actual: {self._actual_width}x{self._actual_height} @ ~{self._reported_fps} reported FPS"
            )
            return True

    def _capture_worker(self):
        """
        Continuously drains frames from VideoCapture to prevent OpenCV DirectShow buffer accumulation.
        Maintains ONLY the latest frame in a single-slot buffer (size=1).
        Discards stale frames when AI processing is busy.
        """
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
        if not self.is_connected:
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
        if self._frame_count == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return round(self._frame_count / elapsed, 2)

    def connect(self) -> bool:
        """Explicit connect method to check or re-establish connection."""
        if self.is_connected and self.cap and self.cap.isOpened():
            return True
        return self._initialize_camera()

    def disconnect(self) -> None:
        """Alias for release to cleanly shut down camera."""
        self.release()

    def release(self) -> None:
        """Stops capture worker thread, releases VideoCapture, and resets state."""
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

            self.is_connected = False
            logger.info(f"SmartBoardCamera index {self.camera_index} released.")

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
