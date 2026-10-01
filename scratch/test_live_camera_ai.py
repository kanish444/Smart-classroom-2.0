import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera.camera_manager import CameraManager
from core.tracking_pipeline import TrackingPipeline
from core.detector import get_face_detector
from core.recognizer import FaceRecognizer
from database.db_manager import DatabaseManager
from core.vector_store import FaissVectorStore
from core.face_embedder import ArcFaceEmbedder

def test_live_camera_pipeline():
    mgr = CameraManager(source="pc")
    print(f"1. Camera Device: Index {mgr.settings.camera.pc.device_index} (Integrated/PC Camera)")
    print(f"2. Camera Connected: {bool(mgr.is_connected)}")
    res = mgr.get_resolution()
    print(f"3. Actual Resolution: {res[0]}x{res[1]}")
    fps = mgr.get_fps()
    print(f"4. Actual/Reported FPS: {fps}")

    success, frame = mgr.read_frame()
    print(f"5. Frame Read Success: {success}, Frame Shape: {frame.shape if success else None}")

    if success:
        detector = get_face_detector()
        print(f"6. Detector Model: {type(detector).__name__}")
        embedder = ArcFaceEmbedder()
        v_store = FaissVectorStore()
        recognizer = FaceRecognizer(embedder=embedder, vector_store=v_store)
        pipeline = TrackingPipeline(detector=detector, recognizer=recognizer)

        t0 = time.perf_counter()
        tracks = pipeline.process_frame(frame)
        total_time = (time.perf_counter() - t0) * 1000.0

        metrics = pipeline.latest_latency_metrics
        print(f"7. Pipeline Latency: Total={total_time:.2f}ms | Detector={metrics.get('detector_ms', 0):.2f}ms | Tracking={metrics.get('tracking_ms', 0):.2f}ms | Recognition={metrics.get('recognition_ms', 0):.2f}ms")
        print(f"8. Raw Faces Detected: {len(pipeline.latest_raw_detections)}")
        print(f"9. Tracked Faces: {len(tracks)}")
        for idx, t in enumerate(tracks):
            student_id = t.stable_student_id or "UNKNOWN"
            status = getattr(t, "display_status", "UNKNOWN")
            print(f"   Face #{idx+1}: Track ID {t.track_id}, Box: {t.bbox}, Identity: {student_id}, State: {status}")

    status_before_stop = mgr.get_status()
    print(f"10. Status Before Stop: {status_before_stop['status']}")

    mgr.disconnect()
    status_after_stop = mgr.get_status()
    print(f"11. Status After Stop: {status_after_stop['status']}")
    print(f"12. Hardware Released: {status_after_stop['connected'] == False}")

if __name__ == "__main__":
    test_live_camera_pipeline()
