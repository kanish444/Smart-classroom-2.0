from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict


class BooleanCallable(int):
    """
    Enables connection state to be queried as a boolean attribute (e.g. `if cam.is_connected:`)
    or invoked as a method (e.g. `if cam.is_connected():`) without raising a TypeError.
    """
    def __call__(self) -> bool:
        return bool(self)

    def __bool__(self) -> bool:
        return super().__bool__()


class BaseCamera(ABC):
    """
    Abstract base class for all camera input sources.
    This ensures that the face recognition engine is decoupled from the camera hardware.
    """

    @abstractmethod
    def get_frame(self) -> Tuple[bool, Any]:
        """
        Reads the next frame from the camera.
        Returns:
            Tuple[bool, Any]: A boolean indicating success, and the frame data (e.g., numpy array).
        """
        pass

    def read_frame(self) -> Tuple[bool, Any]:
        """Alias for get_frame to satisfy frame interface."""
        return self.get_frame()

    @abstractmethod
    def release(self) -> None:
        """
        Releases the camera resources.
        """
        pass

    def disconnect(self) -> None:
        """Alias for release to cleanly shut down camera."""
        self.release()

    def connect(self) -> bool:
        """Explicit connect method to check or establish connection."""
        return bool(self.is_connected)

    @property
    @abstractmethod
    def resolution(self) -> Tuple[int, int]:
        """
        Returns the camera resolution (width, height).
        """
        pass

    def get_resolution(self) -> Tuple[int, int]:
        """Returns the actual camera resolution (width, height)."""
        return self.resolution

    def get_fps(self) -> float:
        """Returns the actual or reported camera frames per second."""
        return getattr(self, "reported_fps", 0.0)

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status dictionary of the camera."""
        is_conn = bool(self.is_connected)
        res = self.resolution
        return {
            "status": "CONNECTED" if is_conn else "DISCONNECTED",
            "connected": is_conn,
            "resolution": f"{res[0]}x{res[1]}",
            "fps": self.get_fps()
        }

    @property
    def is_connected(self) -> BooleanCallable:
        """Returns connection state as BooleanCallable."""
        return BooleanCallable(1 if getattr(self, "_is_connected", False) else 0)

    @is_connected.setter
    def is_connected(self, value: Any):
        self._is_connected = bool(value)
