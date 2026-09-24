import cv2
import time
import numpy as np
from typing import Tuple, Any, Optional
from loguru import logger
from .base_camera import BaseCamera

class SmartBoardCamera(BaseCamera):
    """
    Camera implementation for the Smart Board built-in camera.
    Includes robust frame validation and connection handling.
    """
    def __init__(self, camera_index: int = 0, width: int = 1920, height: int = 1080):
        self.camera_index = camera_index
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
        
        self._initialize_camera()

    def _initialize_camera(self) -> bool:
        logger.info(f"Initializing SmartBoard Camera at index {self.camera_index}...")
        
        # Try standard OpenCV backend first
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            logger.warning(f"Failed to open camera index {self.camera_index}. Trying DirectShow...")
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            
        if not self.cap.isOpened():
            logger.error(f"Could not open camera {self.camera_index}.")
            self.is_connected = False
            return False
            
        # Try to set requested resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
        
        # Read back actual properties
        self._actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera initialized. Actual: {self._actual_width}x{self._actual_height} @ {self._reported_fps} reported FPS")
        self.is_connected = True
        
        # Reset FPS counters
        self._frame_count = 0
        self._start_time = time.time()
        
        return True

    def _validate_frame(self, frame: Any) -> bool:
        """Basic generic frame validation"""
        if frame is None:
            return False
        if not isinstance(frame, np.ndarray):
            return False
        if frame.size == 0:
            return False
        # Check if frame matches expected dimensions
        if frame.shape[0] != self._actual_height or frame.shape[1] != self._actual_width:
            # Not strictly an error if camera changes dynamically, but typically bad
            pass
        return True

    def get_frame(self) -> Tuple[bool, Any]:
        if not self.is_connected or not self.cap or not self.cap.isOpened():
            return False, None
            
        try:
            ret, frame = self.cap.read()
        except Exception as e:
            logger.error(f"Exception while reading frame: {e}")
            ret, frame = False, None
            
        if not ret or not self._validate_frame(frame):
            logger.warning("Failed to grab a valid frame. Camera may be disconnected.")
            self.is_connected = False
            return False, None
            
        self._frame_count += 1
        return True, frame
        
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
        if self.cap:
            self.cap.release()
            self.is_connected = False
            logger.info("Camera released.")

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._actual_width, self._actual_height)
        
    @property
    def reported_fps(self) -> float:
        return self._reported_fps
