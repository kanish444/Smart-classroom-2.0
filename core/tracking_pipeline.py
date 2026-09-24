import time
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from loguru import logger

from core.detector import BaseDetector, YOLOv8FaceDetector
from core.face_processor import FaceProcessor
from core.byte_tracker import ByteTracker, STrack
from core.recognizer import FaceRecognizer
from core.temporal_stabilizer import TemporalStabilizer
from core.schemas import (
    TrackedFace,
    TrackState,
    RecognitionStatus,
    RecognitionReadyFace,
    RecognitionResult,
    FaceQualityResult
)
from config.settings import get_settings


class TrackingPipeline:
    """
    Phase 7 End-to-End Tracking & Temporal Stabilization Pipeline:
    Camera Frame -> Detector -> Quality/Alignment -> ByteTrack -> Recognizer (throttled) -> Stabilizer -> TrackedFaces
    """

    def __init__(
        self,
        detector: Optional[BaseDetector] = None,
        processor: Optional[FaceProcessor] = None,
        tracker: Optional[ByteTracker] = None,
        recognizer: Optional[FaceRecognizer] = None,
        stabilizer: Optional[TemporalStabilizer] = None
    ):
        self.settings = get_settings()
        self.detector = detector or YOLOv8FaceDetector()
        self.processor = processor or FaceProcessor()
        self.tracker = tracker or ByteTracker()
        self.recognizer = recognizer or FaceRecognizer()
        self.stabilizer = stabilizer or TemporalStabilizer()

        self.frame_count = 0
        self.fps_history = []
        self._last_time = time.perf_counter()

    def reset(self):
        """Reset all tracking and stabilization history."""
        self.frame_count = 0
        self.fps_history.clear()
        self.tracker.reset()
        self.stabilizer.reset()

    def process_frame(
        self,
        frame: np.ndarray,
        detections: Optional[List[Dict[str, Any]]] = None,
        timestamp: Optional[float] = None
    ) -> List[TrackedFace]:
        """
        Executes full Phase 7 pipeline on a single frame.

        Args:
            frame: Input BGR image (1080p, 720p, etc.)
            detections: Optional pre-computed detections (for testing or external detector)
            timestamp: Optional epoch timestamp

        Returns:
            List[TrackedFace]: Output tracks with stabilized identities and bounding boxes
        """
        t0 = time.perf_counter()
        self.frame_count += 1
        ts = timestamp or time.time()

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        # 1. Detection
        if detections is None:
            raw_detections = self.detector.detect(frame)
        else:
            raw_detections = detections
        self.latest_raw_detections = raw_detections


        # 2. ByteTrack Association & State Management
        stracks: List[STrack] = self.tracker.update_tracks(raw_detections, frame)

        # 3. Phase 5 Quality Assessment & 5-point Alignment on Tracked Faces
        tracked_detections = [
            {
                "bbox": t.xyxy,
                "confidence": t.score,
                "track_id": t.track_id,
                "face_area": (t.xyxy[2] - t.xyxy[0]) * (t.xyxy[3] - t.xyxy[1]),
                "keypoints": t.detection_data.get("keypoints", [])
            }
            for t in stracks
        ]

        ready_faces, rejected_faces = self.processor.process(tracked_detections, frame, timestamp=ts)
        ready_map: Dict[int, RecognitionReadyFace] = {}
        for rf in ready_faces:
            try:
                tid = int(rf.quality_metrics.face_id)
                ready_map[tid] = rf
            except (ValueError, TypeError):
                pass

        rejected_map: Dict[int, FaceQualityResult] = {}
        for rj in rejected_faces:
            try:
                tid = int(rj.face_id)
                rejected_map[tid] = rj
            except (ValueError, TypeError):
                pass

        # 4. Recognition & Temporal Stabilization
        tracked_faces: List[TrackedFace] = []
        active_track_ids = set()

        for track in stracks:
            active_track_ids.add(track.track_id)
            qm = ready_map.get(track.track_id) or rejected_map.get(track.track_id)
            quality_status = qm.quality_metrics.quality_status if hasattr(qm, 'quality_metrics') else (
                qm.quality_status if qm else "RECOGNITION_READY"
            )

            # Check CPU recognition throttling
            if self.stabilizer.should_recognize(track.track_id, self.frame_count) and track.track_id in ready_map:
                rf = ready_map[track.track_id]
                rec_result = self.recognizer.recognize_face(rf)
            else:
                # Reuse cached state or default to UNKNOWN if new
                summary = self.stabilizer.get_track_summary(track.track_id)
                rec_result = RecognitionResult(
                    face_id=str(track.track_id),
                    matched_student_id=summary.get("stable_student_id"),
                    matched_student_name=summary.get("stable_student_name"),
                    similarity=summary.get("stable_similarity", 0.0),
                    status=summary.get("stable_status", RecognitionStatus.UNKNOWN),
                    threshold=self.recognizer.threshold,
                    embedding_model=self.recognizer.model_name,
                    processing_time_ms=0.0,
                    bbox=track.xyxy,
                    quality_metrics=qm.quality_metrics if hasattr(qm, 'quality_metrics') else qm
                )

            # Update temporal stabilizer
            stable_id, stable_name, stable_sim, stable_status = self.stabilizer.update_track_recognition(
                track_id=track.track_id,
                frame_id=self.frame_count,
                result=rec_result,
                quality_metrics=qm.quality_metrics if hasattr(qm, 'quality_metrics') else qm,
                timestamp=ts
            )

            tracked_face = TrackedFace(
                track_id=track.track_id,
                bbox=track.xyxy,
                state=track.state,
                score=track.score,
                stable_student_id=stable_id,
                stable_student_name=stable_name,
                current_status=rec_result.status,
                current_similarity=float(stable_sim),
                quality_status=quality_status,
                hits=track.hits,
                age=track.age,
                time_since_update=track.time_since_update,
                last_seen=track.last_seen,
                history_len=len(self.stabilizer.track_histories.get(track.track_id, [])),
                quality_metrics=qm.quality_metrics if hasattr(qm, 'quality_metrics') else qm
            )
            tracked_faces.append(tracked_face)

        # 5. Prune memory for dead tracks
        self.stabilizer.cleanup_tracks(active_track_ids)

        return tracked_faces

    def draw_debug_overlay(
        self,
        frame: np.ndarray,
        tracked_faces: List[TrackedFace],
        fps: float = 0.0
    ) -> np.ndarray:
        """
        Draws visual debug HUD overlay on frame:
        - Bounding box color-coded by status
        - Identification label card showing Track ID, Stable ID, Similarity, Quality, and Track State
        - System summary in top-left
        """
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]

        # Draw Global HUD in top-left
        hud_bg = (20, 20, 20)
        cv2.rectangle(vis_frame, (10, 10), (320, 95), hud_bg, -1)
        cv2.rectangle(vis_frame, (10, 10), (320, 95), (60, 60, 60), 1)

        confirmed_count = sum(1 for tf in tracked_faces if tf.stable_student_id is not None)
        unknown_count = sum(1 for tf in tracked_faces if tf.stable_student_id is None)

        cv2.putText(vis_frame, "SmartClass Vision AI - Phase 7 HUD", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis_frame, f"FPS: {fps:.1f} | Frame: {self.frame_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis_frame, f"Tracks: {len(tracked_faces)} (Matches: {confirmed_count}, Unknown: {unknown_count})", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(vis_frame, "Stabilization: ACTIVE | ByteTrack: ENABLED", (20, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 0), 1, cv2.LINE_AA)

        # Draw Face Cards
        for tf in tracked_faces:
            x1, y1, x2, y2 = tf.bbox

            # Choose Color based on status
            if tf.stable_student_id is not None:
                # Confirmed Match -> Green
                box_color = (0, 220, 0)
            elif tf.quality_status == "LOW_QUALITY":
                # Low Quality -> Gray/Yellow
                box_color = (0, 165, 255)
            else:
                # Unknown -> Orange
                box_color = (0, 120, 255)

            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), box_color, 2)

            # Draw card above or below box
            card_w = 200
            card_h = 70
            card_x1 = max(0, min(x1, w - card_w - 5))
            card_y1 = max(0, y1 - card_h - 5) if y1 - card_h - 5 >= 0 else min(h - card_h - 5, y2 + 5)
            card_x2 = card_x1 + card_w
            card_y2 = card_y1 + card_h

            # Semi-transparent card background
            sub_img = vis_frame[card_y1:card_y2, card_x1:card_x2]
            dark_rect = np.zeros(sub_img.shape, dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 0.3, dark_rect, 0.7, 1.0)
            vis_frame[card_y1:card_y2, card_x1:card_x2] = res
            cv2.rectangle(vis_frame, (card_x1, card_y1), (card_x2, card_y2), box_color, 1)

            # Card texts
            id_text = f"ID: {tf.stable_student_id or 'UNKNOWN'}"
            track_text = f"Track: {tf.track_id} [{tf.state}]"
            sim_text = f"Sim: {tf.current_similarity:.2f} ({tf.current_status})"
            qual_text = f"Qual: {tf.quality_status}"

            cv2.putText(vis_frame, track_text, (card_x1 + 6, card_y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis_frame, id_text, (card_x1 + 6, card_y1 + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255) if tf.stable_student_id else (0, 160, 255), 1, cv2.LINE_AA)
            cv2.putText(vis_frame, sim_text, (card_x1 + 6, card_y1 + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(vis_frame, qual_text, (card_x1 + 6, card_y1 + 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)

        return vis_frame
