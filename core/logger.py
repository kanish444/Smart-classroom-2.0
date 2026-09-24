import os
import sys
from loguru import logger
from config.settings import get_settings

def setup_logger():
    """
    Configures the centralized logging system using loguru.
    Ensures logs are written to both console and a rotating file.
    """
    settings = get_settings()
    
    # Remove default logger
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # File handler
    os.makedirs(settings.log_dir, exist_ok=True)
    log_file = os.path.join(settings.log_dir, "smartclass.log")
    
    logger.add(
        log_file,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="10 days"
    )
    
    return logger
