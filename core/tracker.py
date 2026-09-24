from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseTracker(ABC):
    """
    Abstract base class for bounding box trackers (e.g., ByteTrack).
    """
    
    @abstractmethod
    def update(self, detections: List[Dict[str, Any]], frame: Any = None) -> List[Dict[str, Any]]:
        """
        Update the tracker with new detections.
        
        Args:
            detections: List of detections from the detector (bbox, confidence).
            frame: Optional current image frame (some trackers like DeepSORT use this for appearance embeddings).
            
        Returns:
            List[Dict[str, Any]]: A list of tracked objects, where each dict contains:
                - 'track_id': int
                - 'bbox': [x1, y1, x2, y2]
                - 'confidence': float
        """
        pass

# Re-export ByteTracker for convenient access
def get_tracker(**kwargs):
    from core.byte_tracker import ByteTracker
    return ByteTracker(**kwargs)

