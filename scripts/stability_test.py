import sys
import os
import time
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera.camera_manager import CameraManager

def run_stability_test(duration_minutes: int = 5):
    logger.info("=" * 40)
    logger.info(f"CAMERA STABILITY TEST ({duration_minutes} Minutes)")
    logger.info("=" * 40)
    
    manager = CameraManager()
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    frames_processed = 0
    failures = 0
    max_latency = 0.0
    
    logger.info(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        while time.time() < end_time:
            t0 = time.time()
            success, frame = manager.get_frame(max_retries=3)
            latency = time.time() - t0
            
            if latency > max_latency:
                max_latency = latency
                
            if success:
                frames_processed += 1
            else:
                failures += 1
                logger.error("Failed to retrieve frame during stability test.")
                break # If reconnect fails, it's a hard fail
                
            # Log progress every 60 seconds
            elapsed = time.time() - start_time
            if frames_processed % 300 == 0:
                logger.info(f"Progress: {int(elapsed)}s elapsed. Frames: {frames_processed}. True FPS: {manager.get_status().get('measured_fps')}")
                
    except KeyboardInterrupt:
        logger.warning("Test interrupted by user.")
        
    finally:
        total_time = time.time() - start_time
        status = manager.get_status()
        manager.release()
        
        logger.info("=" * 40)
        logger.info("STABILITY TEST RESULTS")
        logger.info("=" * 40)
        logger.info(f"Total Duration : {total_time:.2f} seconds")
        logger.info(f"Frames Read    : {frames_processed}")
        logger.info(f"Failures/Drops : {failures}")
        logger.info(f"Max Latency    : {max_latency*1000:.2f} ms")
        logger.info(f"Final FPS      : {status.get('measured_fps', 0)}")
        logger.info(f"Final Resol.   : {status.get('resolution', 'Unknown')}")
        logger.info("=" * 40)

if __name__ == "__main__":
    # If a command line arg is passed, use it as duration
    mins = 5
    if len(sys.argv) > 1:
        try:
            mins = int(sys.argv[1])
        except ValueError:
            pass
    run_stability_test(mins)
