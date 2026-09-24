import os
import sys
import time
import cv2
import numpy as np
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.camera_manager import SmartBoardCamera
from core.tracking_pipeline import TrackingPipeline
from core.byte_tracker import ByteTracker
from core.recognizer import FaceRecognizer
from core.temporal_stabilizer import TemporalStabilizer
from config.settings import get_settings


def run_live_tracking():
    """
    Runs live multi-face tracking with temporal stabilization on Smart Board camera.
    Press 'q' to exit.
    """
    settings = get_settings()
    logger.info("Initializing SmartClass Vision AI - Phase 7 Live Tracking...")

    camera = SmartBoardCamera()
    if not camera.connect():
        logger.warning(f"Could not connect to camera index {settings.camera.index}. Using synthetic video feed...")
        use_synthetic = True
    else:
        use_synthetic = False

    pipeline = TrackingPipeline(
        tracker=ByteTracker(),
        recognizer=FaceRecognizer(),
        stabilizer=TemporalStabilizer()
    )

    frame_idx = 0
    t0 = time.time()

    print("\n" + "=" * 60)
    print("LIVE TRACKING STARTED — Press 'q' to stop.")
    print("=" * 60)

    try:
        while True:
            frame_start = time.perf_counter()
            frame_idx += 1

            if not use_synthetic:
                success, frame = camera.get_frame()
                if not success or frame is None:
                    continue
                tracked_faces = pipeline.process_frame(frame)
            else:
                # Synthetic classroom simulation
                from scripts.test_phase7_suite import create_synthetic_classroom_frame
                frame, dets = create_synthetic_classroom_frame(1280, 720, num_faces=6, frame_idx=frame_idx)
                tracked_faces = pipeline.process_frame(frame, detections=dets)

            latency = (time.perf_counter() - frame_start) * 1000.0
            fps = 1000.0 / max(1e-3, latency)

            hud_frame = pipeline.draw_debug_overlay(frame, tracked_faces, fps=fps)
            cv2.imshow("SmartClass Vision AI - Phase 7 Live HUD", hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        if not use_synthetic:
            camera.disconnect()
        cv2.destroyAllWindows()
        logger.info(f"Tracking session closed after {frame_idx} frames.")


if __name__ == "__main__":
    run_live_tracking()
