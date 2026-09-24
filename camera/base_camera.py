from abc import ABC, abstractmethod
from typing import Tuple, Any

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
        
    @abstractmethod
    def release(self) -> None:
        """
        Releases the camera resources.
        """
        pass
        
    @property
    @abstractmethod
    def resolution(self) -> Tuple[int, int]:
        """
        Returns the camera resolution (width, height).
        """
        pass
