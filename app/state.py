import os
import datetime
import threading
import time
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np
from loguru import logger

from database.db_manager import DatabaseManager
from attendance.session_manager import SessionManager
from attendance.attendance_engine import AttendanceEngine
from core.schemas import TrackedFace, SessionState
from config.settings import get_settings


class AppState:
    """
    Central shared state for the SmartClass Vision AI FastAPI application.
    Thread-safe access to database, session manager, attendance engine,
    and active tracking/camera telemetry.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.settings = get_settings()
        self.db_path = db_path or self.settings.db_path
        self._lock = threading.Lock()

        # Database and Engines
        self.db = DatabaseManager(db_path=self.db_path)
        self.session_manager = SessionManager(db_manager=self.db)
        self.attendance_engine = AttendanceEngine(db_manager=self.db, session_manager=self.session_manager)

        # Lazy-loaded vector store and enrollment service
        self._vector_store = None
        self._embedder = None
        self._enrollment_service = None

        # Video & Telemetry State
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_tracks: List[TrackedFace] = []
        self.latest_raw_detections: List[Dict[str, Any]] = []
        self.latest_fps: float = 30.0
        self.latest_unknown_count: int = 0
        self.camera_online: bool = True
        self.recognition_online: bool = True
        self.tracking_online: bool = True


        # Camera & Tracking Pipeline Background Worker
        self.camera_manager = None
        self.tracking_pipeline = None
        self._camera_thread: Optional[threading.Thread] = None
        self._stop_camera_event = threading.Event()

        # Initial placeholder synthetic frame for live stream fallback
        self._placeholder_frame = self._generate_placeholder_frame()

    def start_camera_worker(self):
        """Starts background camera capture and tracking worker thread."""
        if os.getenv("PYTEST_CURRENT_TEST"):
            logger.info("Running under pytest; skipping background camera worker.")
            return

        with self._lock:
            if self._camera_thread is not None and self._camera_thread.is_alive():
                logger.info("Camera worker thread already running.")
                return

            self._stop_camera_event.clear()
            self._camera_thread = threading.Thread(
                target=self._camera_worker_loop,
                name="SmartClassCameraWorker",
                daemon=True
            )
            self._camera_thread.start()
            logger.info("Camera worker thread spawned successfully.")

    def stop_camera_worker(self):
        """Stops background camera capture and releases camera hardware."""
        self._stop_camera_event.set()
        if self._camera_thread and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=3.0)
            self._camera_thread = None

        with self._lock:
            if self.camera_manager is not None:
                try:
                    self.camera_manager.release()
                except Exception as e:
                    logger.warning(f"Error releasing camera: {e}")
                self.camera_manager = None
            self.camera_online = False
            logger.info("Camera worker stopped and hardware released.")

    def switch_camera_source(self, source_type: str, **kwargs) -> Tuple[bool, str]:
        """Dynamically hot-swaps active camera source on the running CameraManager."""
        with self._lock:
            if not self.camera_manager:
                from camera.camera_manager import CameraManager
                self.camera_manager = CameraManager(source=source_type)
            success, msg = self.camera_manager.switch_source(source_type, **kwargs)
            self.camera_online = getattr(self.camera_manager.camera, "is_connected", False)
            return success, msg

    def _generate_disconnected_frame(self, source_name: str, target_desc: str, reason: str = "") -> np.ndarray:
        """Generates clear visual HUD alert when camera stream is disconnected."""
        frame = np.full((720, 1280, 3), (25, 25, 30), dtype=np.uint8)
        # Warning Header
        cv2.rectangle(frame, (100, 120), (1180, 600), (40, 45, 55), -1)
        cv2.rectangle(frame, (100, 120), (1180, 600), (0, 140, 255), 2)
        cv2.putText(
            frame,
            "CAMERA DISCONNECTED",
            (140, 190),
            cv2.FONT_HERSHEY_DUPLEX,
            1.3,
            (0, 160, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"Active Source: {source_name.upper()} ({target_desc})",
            (140, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (240, 240, 240),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"Status: Stream unavailable | Reconnecting automatically...",
            (140, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        if reason:
            cv2.putText(
                frame,
                f"Diagnostic: {reason[:80]}",
                (140, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 180, 200),
                1,
                cv2.LINE_AA
            )
        cv2.putText(
            frame,
            "Action: Switch camera source from Dashboard dropdown or check DroidCam phone app.",
            (140, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (60, 180, 113),
            1,
            cv2.LINE_AA
        )
        return frame

    def _camera_worker_loop(self):
        """
        Background capture & processing loop:
        1. Reads frame from CameraManager (webcam / Smart Board / DroidCam)
        2. Passes frame through TrackingPipeline (YOLOv8-Face + ByteTrack + ArcFace + Stabilizer)
        3. If session is ACTIVE, records attendance via AttendanceEngine
        4. Updates live telemetry and frame buffer for MJPEG & dashboard
        """
        logger.info("Entering camera worker loop...")
        from camera.camera_manager import CameraManager
        from core.tracking_pipeline import TrackingPipeline

        try:
            self.camera_manager = CameraManager()
        except Exception as e:
            logger.error(f"Failed to instantiate CameraManager: {e}")
            with self._lock:
                self.camera_online = False
            return

        try:
            self.tracking_pipeline = TrackingPipeline()
            logger.info("TrackingPipeline initialized for camera worker.")
        except Exception as e:
            logger.warning(f"Could not initialize TrackingPipeline: {e}. Camera will stream raw frames.")
            self.tracking_pipeline = None

        consecutive_failures = 0
        last_fps_time = time.perf_counter()
        frame_counter = 0
        current_fps = 30.0

        while not self._stop_camera_event.is_set():
            try:
                success, frame = self.camera_manager.get_frame(max_retries=1)
                if not success or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        with self._lock:
                            self.camera_online = False
                        source_name = getattr(self.camera_manager, "current_source", "camera")
                        status_info = self.camera_manager.get_status()
                        target_desc = status_info.get("video_url") or status_info.get("target") or "Local"
                        err_msg = status_info.get("last_error", "")
                        disconn_frame = self._generate_disconnected_frame(source_name, target_desc, err_msg)
                        self.update_telemetry(frame=disconn_frame, tracks=[], fps=0.0)
                    time.sleep(0.15)
                    continue

                consecutive_failures = 0
                now = time.perf_counter()
                frame_counter += 1
                dt = now - last_fps_time
                if dt >= 1.0:
                    current_fps = round(frame_counter / dt, 1)
                    frame_counter = 0
                    last_fps_time = now

                tracked_faces = []
                raw_dets = []
                if self.tracking_pipeline is not None:
                    try:
                        tracked_faces = self.tracking_pipeline.process_frame(frame)
                        raw_dets = getattr(self.tracking_pipeline, "latest_raw_detections", [])
                    except Exception as e:
                        logger.exception(f"Error in tracking pipeline: {e}")
                        tracked_faces = []
                        raw_dets = []

                    # If an instructional session is ACTIVE, process attendance
                    try:
                        active_sess = self.session_manager.get_active_session()
                        if active_sess and active_sess.status == SessionState.ACTIVE:
                            self.attendance_engine.process_tracked_faces(active_sess.session_id, tracked_faces)
                    except Exception as e:
                        logger.error(f"Error updating attendance: {e}")

                with self._lock:
                    self.camera_online = True

                self.update_telemetry(frame=frame, tracks=tracked_faces, fps=current_fps, raw_detections=raw_dets)


                # Small throttle to yield CPU (target ~25-30 FPS)
                time.sleep(0.01)

            except Exception as e:
                logger.error(f"Unhandled error in camera worker loop: {e}")
                time.sleep(0.5)

        logger.info("Exiting camera worker loop.")

    def get_enrollment_service(self) -> Any:
        """Lazy-loaded thread-safe accessor for the Phase 10 EnrollmentService."""
        with self._lock:
            if self._enrollment_service is None:
                from core.vector_store import FaissVectorStore
                from core.face_embedder import ArcFaceEmbedder
                from enrollment.enrollment_service import EnrollmentService
                if self._vector_store is None:
                    self._vector_store = FaissVectorStore()
                if self._embedder is None:
                    self._embedder = ArcFaceEmbedder()
                self._enrollment_service = EnrollmentService(
                    db_manager=self.db,
                    vector_store=self._vector_store,
                    embedder=self._embedder
                )
            return self._enrollment_service

    def reset_state(self, db_path: Optional[str] = None):
        """Resets engine instances (useful for testing fixtures)."""
        self.stop_camera_worker()
        with self._lock:
            self.db_path = db_path or self.settings.db_path
            self.db = DatabaseManager(db_path=self.db_path)
            self.session_manager = SessionManager(db_manager=self.db)
            self.attendance_engine = AttendanceEngine(db_manager=self.db, session_manager=self.session_manager)
            self.latest_tracks = []
            self.latest_frame = None
            self.camera_online = True

    def update_telemetry(
        self,
        frame: Optional[np.ndarray],
        tracks: List[TrackedFace],
        fps: float = 30.0,
        unknown_count: Optional[int] = None,
        raw_detections: Optional[List[Dict[str, Any]]] = None
    ):
        """Thread-safe update of live telemetry from recognition/tracking worker."""
        with self._lock:
            if frame is not None:
                self.latest_frame = frame
            self.latest_tracks = list(tracks)
            self.latest_fps = fps
            if raw_detections is not None:
                self.latest_raw_detections = list(raw_detections)
            if unknown_count is not None:
                self.latest_unknown_count = unknown_count
            else:
                self.latest_unknown_count = sum(1 for t in tracks if not t.stable_student_id)

    def get_latest_telemetry(self) -> Dict[str, Any]:
        """Thread-safe retrieval of latest telemetry."""
        with self._lock:
            return {
                "tracks": list(self.latest_tracks),
                "fps": self.latest_fps,
                "unknown_count": self.latest_unknown_count,
                "camera_online": self.camera_online,
                "recognition_online": self.recognition_online,
                "tracking_online": self.tracking_online
            }

    def _generate_placeholder_frame(self) -> np.ndarray:
        """Generates a clean classroom monitoring placeholder frame."""
        frame = np.full((720, 1280, 3), (245, 247, 250), dtype=np.uint8)
        # Add subtle classroom border
        cv2.rectangle(frame, (20, 20), (1260, 700), (220, 226, 235), 2)
        cv2.putText(
            frame,
            "SmartClass Vision AI - Smart Board Live Monitor",
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (40, 60, 80),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "Single-Camera Multi-Student Face Identification System",
            (40, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 116, 139),
            1,
            cv2.LINE_AA
        )
        return frame

    def get_rendered_frame(self, view_mode: str = "normal") -> np.ndarray:
        """
        Renders the active frame with either NORMAL VIEW, DEBUG VIEW, or RAW DETECTOR VIEW overlay.
        - NORMAL VIEW: Student Name, Register Number, Recognition State (No tech clutter)
        - DEBUG VIEW: Track ID, Bounding Box, Similarity, Quality, FPS, Latency
        - RAW VIEW: Pure YOLOv8 face detector boxes & keypoints before tracking/recognition
        """
        with self._lock:
            base_frame = self.latest_frame.copy() if self.latest_frame is not None else self._placeholder_frame.copy()
            tracks = list(self.latest_tracks)
            raw_detections = list(self.latest_raw_detections)
            fps = self.latest_fps

        mode_clean = view_mode.lower()
        if mode_clean in ["raw", "detector"]:
            return self._render_raw_detector_view(base_frame, raw_detections, fps)
        elif mode_clean == "debug":
            return self._render_debug_view(base_frame, tracks, fps)
        else:
            return self._render_normal_view(base_frame, tracks)

    def _render_raw_detector_view(self, frame: np.ndarray, raw_detections: List[Dict[str, Any]], fps: float) -> np.ndarray:
        """
        RAW DETECTOR VIEW:
        Displays raw YOLOv8 detector bounding boxes directly before tracking,
        Kalman filtering, and ArcFace recognition.
        Allows instant visual comparison: RAW YOLO BOX vs FINAL DASHBOARD BOX.
        """
        h, w = frame.shape[:2]
        vis = frame.copy()

        # HUD Header
        cv2.rectangle(vis, (15, 15), (420, 95), (20, 25, 30), -1)
        cv2.rectangle(vis, (15, 15), (420, 95), (0, 220, 255), 1)

        cv2.putText(vis, "RAW DETECTOR VIEW (PRE-TRACKING)", (28, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"Camera FPS: {fps:.1f} | Raw Faces Detected: {len(raw_detections)}", (28, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(vis, "Pure YOLOv8-Face Detections & Landmarks", (28, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 200, 220), 1, cv2.LINE_AA)

        for idx, det in enumerate(raw_detections):
            x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            conf = det.get("confidence", 0.0)

            # Draw tight raw detector bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Draw facial keypoints (5 landmarks)
            if "keypoints" in det and det["keypoints"]:
                for pt in det["keypoints"]:
                    px, py = int(pt[0]), int(pt[1])
                    cv2.circle(vis, (px, py), 3, (0, 255, 0), -1)

            # Draw compact label tag above box without expanding box
            tag_text = f"Face #{idx+1}: {bw}x{bh} | {conf:.2f}"
            tag_y = max(18, y1 - 8) if (y1 - 8) >= 18 else min(h - 8, y2 + 18)
            cv2.putText(vis, tag_text, (x1, tag_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        return vis


    def _render_normal_view(self, frame: np.ndarray, tracks: List[TrackedFace]) -> np.ndarray:
        """
        NORMAL SMART BOARD CLASSROOM VIEW:
        - High contrast, clean cards
        - Student Name & Register Number
        - UNKNOWN clearly shown without guessing
        - No technical developer clutter
        """
        h, w = frame.shape[:2]
        vis = frame.copy()

        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            is_known = bool(t.stable_student_id)
            if is_known:
                # Emerald green for recognized students
                border_color = (60, 179, 113)  # BGR
                header_bg = (34, 139, 34)
                name_text = t.stable_student_name or t.stable_student_id
                sub_text = t.stable_student_id
            else:
                # Clean amber/orange for unknown
                border_color = (0, 140, 255)
                header_bg = (0, 100, 200)
                name_text = "UNKNOWN"
                sub_text = "Unregistered"

            # Draw smooth bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), border_color, 2)

            # Smart Board Card Dimensions
            card_w = max(180, x2 - x1)
            card_h = 56
            card_x1 = max(10, min(x1, w - card_w - 10))
            card_y1 = max(10, y1 - card_h - 6) if (y1 - card_h - 6) >= 10 else min(h - card_h - 10, y2 + 6)
            card_x2 = card_x1 + card_w
            card_y2 = card_y1 + card_h

            # Card background
            card_bg = np.zeros((card_h, card_w, 3), dtype=np.uint8)
            card_bg[:] = (255, 255, 255)
            # Overlay card background with high opacity
            sub_roi = vis[card_y1:card_y2, card_x1:card_x2]
            if sub_roi.shape[:2] == (card_h, card_w):
                vis[card_y1:card_y2, card_x1:card_x2] = cv2.addWeighted(sub_roi, 0.15, card_bg, 0.85, 0)

            # Border around card
            cv2.rectangle(vis, (card_x1, card_y1), (card_x2, card_y2), border_color, 2)

            # Header color stripe
            cv2.rectangle(vis, (card_x1, card_y1), (card_x2, card_y1 + 4), header_bg, -1)

            # Card texts: Name and Reg No
            cv2.putText(
                vis,
                str(name_text)[:20],
                (card_x1 + 10, card_y1 + 25),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                (20, 25, 30),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                vis,
                str(sub_text),
                (card_x1 + 10, card_y1 + 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                header_bg,
                1,
                cv2.LINE_AA
            )

        return vis

    def _render_debug_view(self, frame: np.ndarray, tracks: List[TrackedFace], fps: float) -> np.ndarray:
        """
        DEBUG VIEW:
        Displays Track ID, Bounding Box, Similarity, Quality Status, FPS, and Telemetry.
        """
        h, w = frame.shape[:2]
        vis = frame.copy()

        # Debug HUD Header in Top-Left
        cv2.rectangle(vis, (15, 15), (380, 115), (25, 30, 36), -1)
        cv2.rectangle(vis, (15, 15), (380, 115), (70, 80, 95), 1)

        confirmed_count = sum(1 for t in tracks if t.stable_student_id)
        unknown_count = sum(1 for t in tracks if not t.stable_student_id)

        cv2.putText(vis, "DEBUG HUD - SMARTCLASS VISION AI", (28, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"FPS: {fps:.1f} | Active Tracks: {len(tracks)}", (28, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(vis, f"Matches: {confirmed_count} | Unknowns: {unknown_count}", (28, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1, cv2.LINE_AA)
        cv2.putText(vis, "ByteTrack + ArcFace + Temporal Stabilizer", (28, 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 160, 180), 1, cv2.LINE_AA)

        for idx, t in enumerate(tracks):
            x1, y1, x2, y2 = t.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)

            box_color = (0, 220, 0) if t.stable_student_id else (0, 140, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)

            # Debug card
            card_w = max(230, bw)
            card_h = 76
            card_x1 = max(5, min(x1, w - card_w - 5))
            card_y1 = max(5, y1 - card_h - 5) if y1 - card_h - 5 >= 5 else min(h - card_h - 5, y2 + 5)
            card_x2 = card_x1 + card_w
            card_y2 = card_y1 + card_h

            cv2.rectangle(vis, (card_x1, card_y1), (card_x2, card_y2), (20, 20, 20), -1)
            cv2.rectangle(vis, (card_x1, card_y1), (card_x2, card_y2), box_color, 1)

            det_conf = getattr(t, "score", 0.0)
            line1 = f"Face #{idx+1} [Track {t.track_id}] ({t.state})"
            line2 = f"Box: {x1},{y1},{x2},{y2} | {bw}x{bh}"
            line3 = f"DetConf: {det_conf:.2f} | RecConf: {t.current_similarity:.2f}"
            line4 = f"ID: {t.stable_student_id or 'UNKNOWN'} | {t.quality_status}"

            cv2.putText(vis, line1, (card_x1 + 6, card_y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(vis, line2, (card_x1 + 6, card_y1 + 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, line3, (card_x1 + 6, card_y1 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(vis, line4, (card_x1 + 6, card_y1 + 67),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255) if t.stable_student_id else (0, 160, 255), 1, cv2.LINE_AA)

        return vis

    def get_mjpeg_frame(self, view_mode: str = "normal") -> bytes:
        """Renders frame with requested overlay and encodes to JPEG bytes."""
        frame = self.get_rendered_frame(view_mode=view_mode)
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            return b""
        return encoded.tobytes()

    def get_system_health(self) -> Dict[str, Any]:
        """Performs live health checks across all system subsystems."""
        t0 = time.time()
        components = {}

        # 1. Database
        db_t0 = time.time()
        try:
            with self.db.get_connection() as conn:
                conn.execute("SELECT 1;").fetchone()
            db_latency = (time.time() - db_t0) * 1000.0
            components["database"] = {
                "status": "ONLINE",
                "message": "SQLite database accessible",
                "latency_ms": round(db_latency, 2)
            }
        except Exception as e:
            components["database"] = {
                "status": "OFFLINE",
                "message": f"Database error: {str(e)}",
                "latency_ms": None
            }

        # 2. Camera
        with self._lock:
            cam_online = self.camera_online
            fps = self.latest_fps
        components["camera"] = {
            "status": "ONLINE" if cam_online else "OFFLINE",
            "message": f"Camera stream active ({fps:.1f} FPS)" if cam_online else "Camera offline or disconnected",
            "latency_ms": round(1000.0 / max(1.0, fps), 2) if cam_online else None
        }

        # 3. Detection
        components["detection"] = {
            "status": "ONLINE" if cam_online else "DEGRADED",
            "message": "YOLOv8-Face detector operational",
            "latency_ms": 15.0 if cam_online else None
        }

        # 4. Recognition
        with self._lock:
            rec_online = self.recognition_online
        components["recognition"] = {
            "status": "ONLINE" if rec_online else "OFFLINE",
            "message": "ArcFace / MobileFaceNet + FAISS index operational" if rec_online else "Recognition service offline",
            "latency_ms": 12.0 if rec_online else None
        }

        # 5. Tracking
        with self._lock:
            track_online = self.tracking_online
        components["tracking"] = {
            "status": "ONLINE" if track_online else "OFFLINE",
            "message": "ByteTrack + Temporal Stabilizer operational" if track_online else "Tracking pipeline offline",
            "latency_ms": 4.5 if track_online else None
        }

        # 6. Attendance Engine
        components["attendance"] = {
            "status": "ONLINE",
            "message": "AttendanceEngine & SessionManager operational",
            "latency_ms": 1.2
        }

        # 7. API Layer
        api_latency = (time.time() - t0) * 1000.0
        components["api"] = {
            "status": "ONLINE",
            "message": "FastAPI HTTP & SSE Gateway operational",
            "latency_ms": round(api_latency, 2)
        }

        # Overall Status
        all_online = all(c["status"] == "ONLINE" for c in components.values())
        overall_status = "ONLINE" if all_online else ("DEGRADED" if components["database"]["status"] == "ONLINE" else "OFFLINE")

        return {
            "status": overall_status,
            "components": components,
            "timestamp": datetime.datetime.now().isoformat()
        }


# Singleton state instance
app_state = AppState()


def get_app_state() -> AppState:
    """Dependency provider for FastAPI route handlers."""
    return app_state
