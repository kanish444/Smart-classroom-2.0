import sys
import os
import cv2
from loguru import logger
from config.settings import get_settings

def validate_environment() -> bool:
    """
    Validates the Python environment, required directories, and camera availability.
    Returns True if valid, False otherwise.
    """
    settings = get_settings()
    is_valid = True
    
    logger.info("Starting environment validation...")

    # 1. Check Python version
    if sys.version_info < (3, 9):
        logger.error(f"Python version {sys.version_info.major}.{sys.version_info.minor} is unsupported. Require >= 3.9.")
        is_valid = False
    else:
        logger.info(f"Python version check passed: {sys.version_info.major}.{sys.version_info.minor}")

    # 2. Check required directories
    required_dirs = ['app', 'camera', 'core', 'database', 'config', 'utils', 'logs', 'models/weights', 'sample_data']
    for d in required_dirs:
        if not os.path.exists(d):
            logger.error(f"Required directory '{d}' is missing.")
            is_valid = False
        else:
            logger.debug(f"Directory check passed: {d}")

    # 3. Check Camera availability (without grabbing a frame to avoid locking)
    # We just try to open and immediately release
    cap = cv2.VideoCapture(settings.camera.index)
    if not cap.isOpened():
        # Fallback to DSHOW for Windows
        cap = cv2.VideoCapture(settings.camera.index, cv2.CAP_DSHOW)
        
    if not cap.isOpened():
        logger.error(f"Smart Board camera (index {settings.camera.index}) was not detected.")
        logger.error("Suggested action: Check whether Windows recognizes the Smart Board camera.")
        is_valid = False
    else:
        logger.info(f"Camera availability check passed on index {settings.camera.index}.")
        cap.release()

    if is_valid:
        logger.info("Environment validation successful.")
    else:
        logger.error("Environment validation failed.")
        
    return is_valid
