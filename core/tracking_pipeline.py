import time
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from loguru import logger

from core.detector import BaseDetector, YOLOv8FaceDetector, get_face_detector, SCRFDDetector
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
    Phase 7 End-to-End Tracking & Temporal Stabilization Pipeline with Small-Face Support:
    Camera Frame -> Detector (SCRFD ONNX) -> Quality/Alignment -> ByteTrack ->
    Recognizer (throttled) -> Temporal Stabilizer -> TrackedFaces
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
        self.detector = detector or get_face_detector()
        self.processor = processor or FaceProcessor()
        self.tracker = tracker or ByteTracker()
        self.recognizer = recognizer or FaceRecognizer()
        self.stabilizer = stabilizer or TemporalStabilizer()

        self.frame_count = 0
        self.fps_history = []
        self._last_time = time.perf_counter()

        # Telemetry & Performance Metrics (Requirement 12)
        self.latest_latency_metrics: Dict[str, float] = {
            "detector_ms": 0.0,
            "tracking_ms": 0.0,
            "recognition_ms": 0.0,
            "total_ms": 0.0
        }
        self.latest_state_counts: Dict[str, int] = {
            "detected": 0,
            "recognized": 0,
            "verifying": 0,
            "unknown": 0,
            "too_small": 0,
            "low_quality": 0
        }
        self.latest_raw_detections: List[Dict[str, Any]] = []

    def reset(self):
        """Reset all tracking and stabilization history."""
        self.frame_count = 0
        self.fps_history.clear()
        self.tracker.reset()
        self.stabilizer.reset()
        self.latest_raw_detections.clear()

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

        # 1. Detection (Full-frame + controlled tiling recovery)
        t_det_0 = time.perf_counter()
        if detections is None:
            raw_detections = self.detector.detect(frame)
        else:
            raw_detections = detections
        t_det_1 = time.perf_counter()
        det_latency_ms = (t_det_1 - t_det_0) * 1000.0
        self.latest_raw_detections = raw_detections

        # 2. ByteTrack Association & State Management
        t_track_0 = time.perf_counter()
        stracks: List[STrack] = self.tracker.update_tracks(raw_detections, frame)
        t_track_1 = time.perf_counter()
        track_latency_ms = (t_track_1 - t_track_0) * 1000.0

        # 3. Quality Assessment & 5-point Alignment on Tracked Faces
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
        t_rec_0 = time.perf_counter()
        tracked_faces: List[TrackedFace] = []
        active_track_ids = set()

        cnt_rec = 0
        cnt_ver = 0
        cnt_unk = 0
        cnt_small = 0
        cnt_lq = 0

        for track in stracks:
            active_track_ids.add(track.track_id)
            qm = ready_map.get(track.track_id) or rejected_map.get(track.track_id)
            quality_status = qm.quality_metrics.quality_status if hasattr(qm, 'quality_metrics') else (
                qm.quality_status if qm else "RECOGNITION_READY"
            )

            # Check CPU recognition throttling (force fresh embedding if track was temporarily lost/reassociated)
            was_reassociated = getattr(track, 'time_since_update', 0) > 1
            if self.stabilizer.should_recognize(track.track_id, self.frame_count, force_recheck=was_reassociated) and track.track_id in ready_map:
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

            # Categorize user-facing display status (Requirements 5, 9, 10, 11)
            diag_qm = qm.quality_metrics if hasattr(qm, 'quality_metrics') else qm
            rej_reason = (getattr(diag_qm, "rejection_reason", None) or "").lower() if diag_qm else ""

            if stable_id is not None:
                disp_status = "RECOGNIZED"
                cnt_rec += 1
            elif quality_status == "FACE_TOO_SMALL" or "too small" in rej_reason:
                disp_status = "FACE TOO SMALL"
                cnt_small += 1
            elif quality_status == "LOW_QUALITY":
                disp_status = "LOW QUALITY"
                cnt_lq += 1
            elif stable_status == "VERIFYING":
                disp_status = "VERIFYING"
                cnt_ver += 1
            else:
                disp_status = "UNKNOWN"
                cnt_unk += 1

            # Build diagnostic telemetry record (Requirements 12 & 15)
            diag_qm = qm.quality_metrics if hasattr(qm, 'quality_metrics') else qm
            face_w = track.xyxy[2] - track.xyxy[0]
            face_h = track.xyxy[3] - track.xyxy[1]
            diagnostics = {
                "face_size": f"{face_w}x{face_h}",
                "face_area": face_w * face_h,
                "brightness": getattr(diag_qm, "brightness", 0.0) if diag_qm else 0.0,
                "sharpness": getattr(diag_qm, "sharpness", 0.0) if diag_qm else 0.0,
                "contrast": getattr(diag_qm, "contrast", 0.0) if diag_qm else 0.0,
                "glare_score": getattr(diag_qm, "glare_score", 0.0) if diag_qm else 0.0,
                "detection_confidence": round(float(track.score), 2),
                "recognition_similarity": round(float(stable_sim), 2),
                "track_id": track.track_id,
                "display_status": disp_status,
                "rejection_reason": getattr(diag_qm, "rejection_reason", None) if diag_qm else None
            }

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
                quality_metrics=diag_qm,
                display_status=disp_status,
                diagnostics=diagnostics
            )
            tracked_faces.append(tracked_face)

        t_rec_1 = time.perf_counter()
        rec_latency_ms = (t_rec_1 - t_rec_0) * 1000.0

        # 5. Prune memory for dead tracks
        self.stabilizer.cleanup_tracks(active_track_ids)

        t_total = (time.perf_counter() - t0) * 1000.0

        # Record telemetry
        self.latest_latency_metrics = {
            "detector_ms": round(det_latency_ms, 1),
            "tracking_ms": round(track_latency_ms, 1),
            "recognition_ms": round(rec_latency_ms, 1),
            "total_ms": round(t_total, 1)
        }
        self.latest_state_counts = {
            "detected": len(raw_detections),
            "recognized": cnt_rec,
            "verifying": cnt_ver,
            "unknown": cnt_unk,
            "too_small": cnt_small,
            "low_quality": cnt_lq
        }

        return tracked_faces

    def draw_debug_overlay(
        self,
        frame: np.ndarray,
        tracked_faces: List[TrackedFace],
        fps: float = 0.0
    ) -> np.ndarray:
        """
        Draws visual debug HUD overlay on frame:
        - GREEN: Recognized student (Name, Reg No, RECOGNIZED)
        - YELLOW/ORANGE: Verifying, Unknown, Low Quality, Face Too Small
        - RED: System/processing error
        - Detailed diagnostic telemetry cards
        """
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]

        # Draw Global HUD in top-left (Requirement 14)
        hud_bg = (20, 24, 30)
        cv2.rectangle(vis_frame, (10, 10), (390, 125), hud_bg, -1)
        cv2.rectangle(vis_frame, (10, 10), (390, 125), (60, 75, 90), 1)

        counts = self.latest_state_counts
        lat = self.latest_latency_metrics
        total_faces = counts.get('detected', 0)
        rec_faces = counts.get('recognized', 0)
        unk_faces = counts.get('unknown', 0)
        unsuitable_faces = counts.get('too_small', 0) + counts.get('low_quality', 0)

        cv2.putText(vis_frame, "SmartClass Vision AI - SCRFD Monitor HUD", (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(vis_frame, f"Cam: {fps:.1f} FPS | Frame: {self.frame_count} | Lat: {lat.get('total_ms', 0):.0f}ms", (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis_frame, f"SCRFD: {lat.get('detector_ms', 0):.1f}ms | Rec: {lat.get('recognition_ms', 0):.1f}ms | Track: {lat.get('tracking_ms', 0):.1f}ms", (20, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 210, 230), 1, cv2.LINE_AA)
        cv2.putText(vis_frame, f"TOTAL FACES: {total_faces}  |  RECOGNIZED: {rec_faces}", (20, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0) if rec_faces > 0 else (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(vis_frame, f"UNKNOWN: {unk_faces}  |  NOT SUITABLE: {unsuitable_faces}", (20, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1, cv2.LINE_AA)

        # Draw Face Cards
        for tf in tracked_faces:
            x1, y1, x2, y2 = tf.bbox
            disp = tf.display_status or ("RECOGNIZED" if tf.stable_student_id else "UNKNOWN")

            # Color coding (Requirement 11)
            if disp == "RECOGNIZED":
                box_color = (0, 220, 0)       # Green
            elif disp == "VERIFYING":
                box_color = (0, 200, 255)     # Yellow / Gold
            elif disp == "FACE TOO SMALL":
                box_color = (0, 165, 255)     # Amber / Orange
            elif disp == "LOW QUALITY":
                box_color = (0, 165, 255)     # Amber / Orange
            else:
                box_color = (0, 120, 255)     # Orange

            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), box_color, 2)

            # Draw card above or below box
            card_w = max(200, x2 - x1)
            card_h = 68
            card_x1 = max(0, min(x1, w - card_w - 5))
            card_y1 = max(0, y1 - card_h - 5) if y1 - card_h - 5 >= 0 else min(h - card_h - 5, y2 + 5)
            card_x2 = card_x1 + card_w
            card_y2 = card_y1 + card_h

            # Semi-transparent card background
            sub_img = vis_frame[card_y1:card_y2, card_x1:card_x2]
            dark_rect = np.zeros(sub_img.shape, dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 0.25, dark_rect, 0.75, 1.0)
            vis_frame[card_y1:card_y2, card_x1:card_x2] = res
            cv2.rectangle(vis_frame, (card_x1, card_y1), (card_x2, card_y2), box_color, 1)

            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)

            # Card texts (Requirement 14)
            if disp == "RECOGNIZED":
                line1 = f"{tf.stable_student_name or tf.stable_student_id}"
                line2 = f"{tf.stable_student_id} | RECOGNIZED"
                line3 = f"Sim: {tf.current_similarity:.2f} | Conf: {tf.score:.2f}"
            elif disp == "VERIFYING":
                line1 = "VERIFYING..."
                line2 = f"Track: {tf.track_id}"
                line3 = f"Conf: {tf.score:.2f}"
            elif disp == "FACE TOO SMALL":
                line1 = "UNKNOWN"
                line2 = "FACE TOO SMALL"
                line3 = f"Size: {bw}x{bh} px"
            elif disp == "LOW QUALITY":
                line1 = "UNKNOWN"
                line2 = "LOW QUALITY"
                line3 = f"Size: {bw}x{bh} px"
            else:
                line1 = "UNKNOWN"
                line2 = f"Track: {tf.track_id}"
                line3 = f"Conf: {tf.score:.2f} | Sim: {tf.current_similarity:.2f}"

            cv2.putText(vis_frame, str(line1)[:22], (card_x1 + 6, card_y1 + 18),
                        cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255) if disp == "RECOGNIZED" else (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(vis_frame, str(line2)[:28], (card_x1 + 6, card_y1 + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 240, 180) if disp == "RECOGNIZED" else (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(vis_frame, str(line3), (card_x1 + 6, card_y1 + 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 160), 1, cv2.LINE_AA)

        return vis_frame
