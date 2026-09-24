import sys
import os
import time
import cv2
import numpy as np
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.camera_manager import CameraManager
from core.detector import YOLOv8FaceDetector
from core.face_processor import FaceProcessor
from core.face_embedder import ArcFaceEmbedder
from database.db_manager import DatabaseManager
from core.vector_store import FaissVectorStore
from core.recognizer import FaceRecognizer
from enrollment.enrollment_service import EnrollmentService
from core.schemas import RecognitionStatus
from config.settings import get_settings


def run_recognition_preview(image_path: str = None):
    """
    Development-only recognition preview interface.
    Overlays Face ID, Similarity, and MATCH / UNKNOWN status.
    Uses sample test identities only.
    """
    settings = get_settings()
    logger.info("Initializing Phase 6 Recognition Preview Interface...")

    db = DatabaseManager()
    store = FaissVectorStore()
    embedder = ArcFaceEmbedder()
    recognizer = FaceRecognizer(embedder=embedder, vector_store=store, threshold=settings.recognition.similarity_threshold)
    processor = FaceProcessor()
    detector = YOLOv8FaceDetector(model_path="yolov8n-face.pt")

    # If database is empty, seed 2 sample test identities for demonstration
    if db.get_student_count() == 0:
        logger.info("Seeding test identities for demonstration...")
        enrollment = EnrollmentService(db, store, embedder)
        enrollment.register_student("TEST_ALPHA", "Sample Subject Alpha")
        enrollment.register_student("TEST_BETA", "Sample Subject Beta")

    window_name = "SmartClass Vision AI — Phase 6 Recognition Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if image_path and os.path.exists(image_path):
        frame = cv2.imread(image_path)
        is_live = False
        manager = None
    else:
        logger.info("Opening camera stream...")
        manager = CameraManager()
        is_live = True

    try:
        while True:
            if is_live:
                success, frame = manager.get_frame(max_retries=3)
                if not success or frame is None:
                    logger.warning("Frame acquisition failed. Exiting.")
                    break
            else:
                frame_to_show = frame.copy()

            t0 = time.perf_counter()
            dets = detector.detect(frame if is_live else frame_to_show)
            faces = dets
            ready_faces, rejected_results = processor.process(faces, frame if is_live else frame_to_show)

            results = recognizer.recognize_batch(ready_faces)
            total_time_ms = (time.perf_counter() - t0) * 1000.0

            display_frame = frame if is_live else frame_to_show

            # Draw recognition results
            for res in results:
                if len(res.bbox) == 4:
                    x1, y1, x2, y2 = res.bbox
                    is_match = (res.status == RecognitionStatus.MATCH)
                    color = (0, 255, 0) if is_match else (0, 0, 255)

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                    if is_match:
                        label = f"{res.matched_student_name or res.matched_student_id} ({res.similarity:.2f})"
                        status_label = "MATCH"
                    else:
                        label = f"UNKNOWN ({res.similarity:.2f})"
                        status_label = "UNKNOWN"

                    cv2.putText(display_frame, status_label, (x1, max(20, y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    cv2.putText(display_frame, label, (x1, max(40, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Draw rejected low-quality faces
            for rej in rejected_results:
                if len(rej.bbox) == 4:
                    x1, y1, x2, y2 = rej.bbox
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 140, 255), 1)
                    cv2.putText(display_frame, f"LOW_Q: {rej.rejection_reason[:15]}", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)

            # Diagnostic HUD
            hud_text = [
                f"Faces Detected: {len(faces)}",
                f"Recognized Matches: {sum(1 for r in results if r.status == RecognitionStatus.MATCH)}",
                f"Unknown Faces: {sum(1 for r in results if r.status == RecognitionStatus.UNKNOWN)}",
                f"Inference Latency: {total_time_ms:.1f} ms",
                f"Threshold: {recognizer.threshold:.2f}"
            ]
            y0 = 30
            for i, line in enumerate(hud_text):
                cv2.putText(display_frame, line, (20, y0 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1 if is_live else 0) & 0xFF
            if key == ord('q') or not is_live:
                break

    finally:
        if manager:
            manager.release()
        cv2.destroyAllWindows()
        logger.info("Recognition preview closed.")


if __name__ == "__main__":
    test_img = sys.argv[1] if len(sys.argv) > 1 else None
    run_recognition_preview(test_img)
