import time
from collections import deque
from typing import Dict, Any, Optional, Tuple, Set, List
from loguru import logger

from core.schemas import (
    RecognitionResult,
    RecognitionStatus,
    FaceQualityResult,
    RecognitionObservation,
    TrackState
)
from config.settings import get_settings


class TemporalStabilizer:
    """
    Temporal Recognition Stabilizer for Multi-Face Tracking:
    - Maintains per-track recognition history in rolling ring buffers
    - Eliminates single-frame recognition flickering (MATCH -> UNKNOWN -> MATCH)
    - Enforces Identity Switch Protection (requires consecutive consistent matches)
    - Provides Quality-Aware resilience (blurry/occluded frames don't drop stable IDs)
    - Preserves strict Unknown Rejection
    - Manages recognition frequency throttling to reduce CPU load
    - Performs memory cleanup when tracks are removed
    """

    def __init__(
        self,
        history_length: Optional[int] = None,
        min_stable_observations: Optional[int] = None,
        identity_switch_threshold: Optional[int] = None,
        unknown_persistence_duration: Optional[int] = None,
        poor_quality_timeout: Optional[int] = None,
        recognition_interval: Optional[int] = None,
        re_embedding_on_low_confidence: Optional[bool] = None
    ):
        settings = get_settings().stabilization
        self.history_length = history_length if history_length is not None else settings.history_length
        self.min_stable_observations = min_stable_observations if min_stable_observations is not None else settings.min_stable_observations
        self.identity_switch_threshold = identity_switch_threshold if identity_switch_threshold is not None else settings.identity_switch_threshold
        self.unknown_persistence_duration = unknown_persistence_duration if unknown_persistence_duration is not None else settings.unknown_persistence_duration
        self.poor_quality_timeout = poor_quality_timeout if poor_quality_timeout is not None else settings.poor_quality_timeout
        self.recognition_interval = recognition_interval if recognition_interval is not None else settings.recognition_interval
        self.re_embedding_on_low_confidence = re_embedding_on_low_confidence if re_embedding_on_low_confidence is not None else settings.re_embedding_on_low_confidence

        # Per-track ring buffers of RecognitionObservation
        self.track_histories: Dict[int, deque] = {}
        # Per-track internal state
        self.track_states: Dict[int, Dict[str, Any]] = {}

    def reset(self):
        """Clears all tracking history and state."""
        self.track_histories.clear()
        self.track_states.clear()

    def _get_track_state(self, track_id: int) -> Dict[str, Any]:
        if track_id not in self.track_states:
            self.track_states[track_id] = {
                "stable_student_id": None,
                "stable_student_name": None,
                "stable_similarity": 0.0,
                "stable_status": RecognitionStatus.UNKNOWN,
                "consecutive_unknown_count": 0,
                "consecutive_candidate_id": None,
                "consecutive_candidate_count": 0,
                "poor_quality_count": 0,
                "last_recognition_frame": -1,
                "last_recognition_time": 0.0,
                "last_raw_status": RecognitionStatus.UNKNOWN,
                "last_raw_similarity": 0.0,
                "total_recognitions": 0
            }
        if track_id not in self.track_histories:
            self.track_histories[track_id] = deque(maxlen=self.history_length)
        return self.track_states[track_id]

    def should_recognize(self, track_id: int, frame_id: int) -> bool:
        """
        Determines whether recognition (ArcFace embedding + FAISS search) should execute
        for this track on the current frame.
        Throttling saves up to 75% CPU by skipping redundant embeddings on stable faces.
        """
        if track_id not in self.track_states:
            return True

        state = self.track_states[track_id]
        # Always recognize if unconfirmed
        if state["stable_student_id"] is None:
            return True

        # Always re-recognize if last similarity was borderline near threshold
        if self.re_embedding_on_low_confidence and state["stable_similarity"] < 0.70:
            return True

        # Periodic check
        if frame_id - state["last_recognition_frame"] >= self.recognition_interval:
            return True

        return False

    def update_track_recognition(
        self,
        track_id: int,
        frame_id: int,
        result: RecognitionResult,
        quality_metrics: Optional[FaceQualityResult] = None,
        timestamp: Optional[float] = None
    ) -> Tuple[Optional[str], Optional[str], float, str]:
        """
        Ingests a new recognition result for a track and returns stabilized identity:
        Returns:
            (stable_student_id, stable_student_name, stable_similarity, stable_status)
        """
        ts = timestamp or time.time()
        state = self._get_track_state(track_id)
        history = self.track_histories[track_id]

        state["last_recognition_frame"] = frame_id
        state["last_recognition_time"] = ts
        state["total_recognitions"] += 1
        state["last_raw_status"] = result.status
        state["last_raw_similarity"] = result.similarity

        # Record observation
        quality_status = "RECOGNITION_READY"
        sharpness = 0.0
        if quality_metrics:
            quality_status = quality_metrics.quality_status
            sharpness = quality_metrics.sharpness
        elif result.quality_metrics:
            quality_status = result.quality_metrics.quality_status
            sharpness = result.quality_metrics.sharpness

        obs = RecognitionObservation(
            frame_id=frame_id,
            timestamp=ts,
            status=result.status,
            student_id=result.matched_student_id,
            student_name=result.matched_student_name,
            similarity=result.similarity,
            quality_status=quality_status,
            sharpness=sharpness
        )
        history.append(obs)

        # -------------------------------------------------------------
        # 1. Quality-Aware Filtering
        # -------------------------------------------------------------
        if quality_status == "LOW_QUALITY":
            state["poor_quality_count"] += 1
            if state["stable_student_id"] is not None:
                if state["poor_quality_count"] < self.poor_quality_timeout:
                    # Maintain established identity during temporary poor-quality frames
                    return (
                        state["stable_student_id"],
                        state["stable_student_name"],
                        state["stable_similarity"],
                        RecognitionStatus.MATCH
                    )
                else:
                    # Poor quality has persisted for too long; degrade to UNKNOWN
                    state["stable_student_id"] = None
                    state["stable_student_name"] = None
                    state["stable_similarity"] = 0.0
                    state["stable_status"] = RecognitionStatus.UNKNOWN
                    return (None, None, 0.0, RecognitionStatus.UNKNOWN)
            else:
                # Poor-quality frame cannot establish a new student match
                state["stable_status"] = RecognitionStatus.UNKNOWN
                return (None, None, 0.0, RecognitionStatus.UNKNOWN)
        else:
            state["poor_quality_count"] = 0

        # -------------------------------------------------------------
        # 2. Process Recognition Outcome
        # -------------------------------------------------------------
        # Case A: UNKNOWN (Similarity < Threshold)
        if result.status == RecognitionStatus.UNKNOWN:
            state["consecutive_unknown_count"] += 1
            state["consecutive_candidate_id"] = None
            state["consecutive_candidate_count"] = 0

            if state["stable_student_id"] is not None:
                # Flickering protection: Bridge short gaps of UNKNOWN
                if state["consecutive_unknown_count"] < self.unknown_persistence_duration:
                    return (
                        state["stable_student_id"],
                        state["stable_student_name"],
                        state["stable_similarity"],
                        RecognitionStatus.MATCH
                    )
                else:
                    # UNKNOWN has persisted beyond allowed duration
                    state["stable_student_id"] = None
                    state["stable_student_name"] = None
                    state["stable_similarity"] = result.similarity
                    state["stable_status"] = RecognitionStatus.UNKNOWN
                    return (None, None, result.similarity, RecognitionStatus.UNKNOWN)
            else:
                state["stable_status"] = RecognitionStatus.UNKNOWN
                return (None, None, result.similarity, RecognitionStatus.UNKNOWN)

        # Case B: MATCH (Similarity >= Threshold)
        elif result.status == RecognitionStatus.MATCH:
            new_id = result.matched_student_id
            new_name = result.matched_student_name
            sim = result.similarity
            state["consecutive_unknown_count"] = 0

            # Subcase B1: Track has no stable identity yet (NEW / UNCONFIRMED)
            if state["stable_student_id"] is None:
                # Count recent observations of new_id in history
                match_count = sum(1 for o in history if o.student_id == new_id and o.status == RecognitionStatus.MATCH)
                if match_count >= self.min_stable_observations:
                    state["stable_student_id"] = new_id
                    state["stable_student_name"] = new_name
                    state["stable_similarity"] = sim
                    state["stable_status"] = RecognitionStatus.MATCH
                    return (new_id, new_name, sim, RecognitionStatus.MATCH)
                else:
                    # Accumulating evidence; hold as UNKNOWN until min_stable_observations reached
                    return (None, None, sim, RecognitionStatus.UNKNOWN)

            # Subcase B2: Matches current stable identity
            elif state["stable_student_id"] == new_id:
                state["consecutive_candidate_id"] = None
                state["consecutive_candidate_count"] = 0
                # Exponential moving average of similarity
                state["stable_similarity"] = 0.7 * state["stable_similarity"] + 0.3 * sim
                state["stable_status"] = RecognitionStatus.MATCH
                return (state["stable_student_id"], state["stable_student_name"], state["stable_similarity"], RecognitionStatus.MATCH)

            # Subcase B3: Matches a DIFFERENT student (Identity Switch Attempt)
            else:
                if state["consecutive_candidate_id"] == new_id:
                    state["consecutive_candidate_count"] += 1
                else:
                    state["consecutive_candidate_id"] = new_id
                    state["consecutive_candidate_count"] = 1

                # Only switch if evidence is overwhelming and sustained
                if state["consecutive_candidate_count"] >= self.identity_switch_threshold:
                    logger.info(
                        f"Identity switch confirmed on Track {track_id}: "
                        f"{state['stable_student_id']} -> {new_id} after {state['consecutive_candidate_count']} frames."
                    )
                    state["stable_student_id"] = new_id
                    state["stable_student_name"] = new_name
                    state["stable_similarity"] = sim
                    state["stable_status"] = RecognitionStatus.MATCH
                    state["consecutive_candidate_id"] = None
                    state["consecutive_candidate_count"] = 0
                    return (new_id, new_name, sim, RecognitionStatus.MATCH)
                else:
                    # Suppress momentary switch and preserve current stable identity
                    return (
                        state["stable_student_id"],
                        state["stable_student_name"],
                        state["stable_similarity"],
                        RecognitionStatus.MATCH
                    )

        # Case C: INVALID or ERROR
        else:
            if state["stable_student_id"] is not None:
                return (
                    state["stable_student_id"],
                    state["stable_student_name"],
                    state["stable_similarity"],
                    RecognitionStatus.MATCH
                )
            return (None, None, 0.0, RecognitionStatus.UNKNOWN)

    def get_track_summary(self, track_id: int) -> Dict[str, Any]:
        """Returns the current temporal stabilization state for a track."""
        state = self._get_track_state(track_id)
        return {
            "track_id": track_id,
            "stable_student_id": state["stable_student_id"],
            "stable_student_name": state["stable_student_name"],
            "stable_similarity": state["stable_similarity"],
            "stable_status": state["stable_status"],
            "last_raw_status": state["last_raw_status"],
            "last_raw_similarity": state["last_raw_similarity"],
            "history_len": len(self.track_histories.get(track_id, []))
        }

    def cleanup_tracks(self, active_track_ids: Set[int]):
        """
        Removes all temporal history and cached state for tracks that are no longer active,
        preventing unbounded memory growth.
        """
        dead_ids = [tid for tid in list(self.track_states.keys()) if tid not in active_track_ids]
        for tid in dead_ids:
            self.track_states.pop(tid, None)
            self.track_histories.pop(tid, None)
