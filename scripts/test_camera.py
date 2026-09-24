import sys
import os
import cv2
from loguru import logger

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.smartboard_camera import SmartBoardCamera

def test_camera():
    logger.info("Starting camera test...")
    try:
        # Default index 0 for laptop webcam or smartboard camera
        cam = SmartBoardCamera(camera_index=0)
        
        logger.info(f"Camera initialized with resolution: {cam.resolution}")
        logger.info("Attempting to capture 5 frames...")
        
        for i in range(5):
            success, frame = cam.get_frame()
            if success:
                logger.info(f"Successfully captured frame {i+1}. Shape: {frame.shape}")
            else:
                logger.error(f"Failed to capture frame {i+1}")
                
        cam.release()
        logger.info("Camera test completed successfully.")
        
    except Exception as e:
        logger.error(f"Camera test failed: {e}")

if __name__ == "__main__":
    test_camera()
