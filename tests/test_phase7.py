import os
import sys
import time
import tempfile
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.byte_tracker import ByteTracker, STrack, KalmanFilterXYAH
from core.temporal_stabilizer import TemporalStabilizer
from core.tracking_pipeline import TrackingPipeline
from core.schemas import (
    TrackState,
    TrackedFace,
    RecognitionResult,
    RecognitionStatus,
    FaceQualityResult,
    RecognitionReadyFace
)
from database.db_manager import DatabaseManager
from core.vector_store import FaissVectorStore
from core.recognizer import FaceRecognizer
from core.face_embedder import ArcFaceEmbedder


def make_det(x1: int, y1: int, x2: int, y2: int, conf: float = 0.9, keypoints=None) -> dict:
    """Helper to construct a standardized detection dictionary."""
    w = x2 - x1
    h = y2 - y1
    if keypoints is None:
        keypoints = [
            [x1 + w * 0.3, y1 + h * 0.4],
            [x1 + w * 0.7, y1 + h * 0.4],
            [x1 + w * 0.5, y1 + h * 0.6],
            [x1 + w * 0.35, y1 + h * 0.8],
            [x1 + w * 0.65, y1 + h * 0.8]
        ]
    return {
        "bbox": [x1, y1, x2, y2],
        "confidence": conf,
        "face_area": w * h,
        "zone": "NEAR" if w * h > 15000 else "MIDDLE",
        "keypoints": keypoints
    }


def make_quality_result(face_id: str, bbox: list, quality_status: str = "RECOGNITION_READY", sharpness: float = 85.0) -> FaceQualityResult:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return FaceQualityResult(
        face_id=face_id,
        bbox=bbox,
        keypoints=[[10, 10], [20, 10], [15, 15], [12, 20], [18, 20]],
        width=w,
        height=h,
        area=w * h,
        sharpness=sharpness,
        brightness=120.0,
        quality_status=quality_status,
        rejection_reason=None if quality_status == "RECOGNITION_READY" else "Low sharpness"
    )


# ===========================================================================
# SCENARIO 1: One Face Continuously Visible
# ===========================================================================
def test_01_one_face_continuously_visible():
    tracker = ByteTracker()
    track_ids = []

    for frame_idx in range(15):
        # Face moves smoothly across frames
        x_offset = frame_idx * 4
        det = make_det(100 + x_offset, 100, 220 + x_offset, 250, conf=0.92)
        tracks = tracker.update([det])

        assert len(tracks) == 1
        track = tracks[0]
        track_ids.append(track["track_id"])
        assert track["confidence"] == pytest.approx(0.92, rel=1e-3)

    # Track ID must remain 100% constant across all frames
    assert len(set(track_ids)) == 1
    assert tracks[0]["state"] == TrackState.ACTIVE


# ===========================================================================
# SCENARIO 2: Multiple Faces Continuously Visible
# ===========================================================================
def test_02_multiple_faces_continuously_visible():
    tracker = ByteTracker()
    history = {0: [], 1: [], 2: []}

    for frame_idx in range(12):
        dets = [
            make_det(100 + frame_idx, 100, 200 + frame_idx, 220, conf=0.91),
            make_det(400, 150 + frame_idx, 510, 280 + frame_idx, conf=0.88),
            make_det(800 - frame_idx, 200, 920 - frame_idx, 340, conf=0.94)
        ]
        tracks = tracker.update(dets)
        assert len(tracks) == 3

        # Sort by x1 position to match across frames
        tracks_sorted = sorted(tracks, key=lambda t: t["bbox"][0])
        for i in range(3):
            history[i].append(tracks_sorted[i]["track_id"])

    # Each position must retain its own distinct track ID throughout
    for i in range(3):
        assert len(set(history[i])) == 1
    all_assigned_ids = {history[0][0], history[1][0], history[2][0]}
    assert len(all_assigned_ids) == 3


# ===========================================================================
# SCENARIO 3: Face Enters Frame
# ===========================================================================
def test_03_face_enters_frame():
    tracker = ByteTracker(min_hits_to_activate=2)

    # Frame 1: Empty frame
    tracks_f1 = tracker.update([])
    assert len(tracks_f1) == 0

    # Frame 2: Face enters frame
    det = make_det(150, 150, 260, 280, conf=0.89)
    tracks_f2 = tracker.update([det])
    assert len(tracks_f2) == 1
    t2 = tracks_f2[0]
    assert t2["state"] == TrackState.NEW

    # Frame 3: Face stays in frame -> transitions to ACTIVE
    tracks_f3 = tracker.update([det])
    assert len(tracks_f3) == 1
    t3 = tracks_f3[0]
    assert t3["track_id"] == t2["track_id"]
    assert t3["state"] == TrackState.ACTIVE


# ===========================================================================
# SCENARIO 4: Face Leaves Frame
# ===========================================================================
def test_04_face_leaves_frame():
    tracker = ByteTracker(max_lost_frames=5)
    det = make_det(200, 200, 310, 330, conf=0.90)

    # Present for 3 frames
    for _ in range(3):
        tracker.update([det])
    assert len(tracker.tracked_stracks) == 1
    tid = tracker.tracked_stracks[0].track_id

    # Face leaves frame (empty detections)
    # Frame 4: transitions to LOST
    t_f4 = tracker.update([])
    assert len(t_f4) == 0
    assert len(tracker.lost_stracks) == 1
    assert tracker.lost_stracks[0].track_id == tid
    assert tracker.lost_stracks[0].state == TrackState.LOST

    # Exceed max_lost_frames (5 frames)
    for _ in range(6):
        tracker.update([])

    # Must be completely removed from active and lost tracks
    assert len(tracker.tracked_stracks) == 0
    assert len(tracker.lost_stracks) == 0
    assert any(t.track_id == tid for t in tracker.removed_stracks)


# ===========================================================================
# SCENARIO 5: Face Temporarily Disappears
# ===========================================================================
def test_05_face_temporarily_disappears():
    tracker = ByteTracker(max_lost_frames=10)
    det = make_det(300, 300, 420, 440, conf=0.95)

    # Active for 4 frames
    for _ in range(4):
        tracker.update([det])
    initial_id = tracker.tracked_stracks[0].track_id

    # Disappears for 2 frames
    tracker.update([])
    tracker.update([])
    assert len(tracker.lost_stracks) == 1

    # Reappears at same/nearby location
    det_return = make_det(305, 305, 425, 445, conf=0.94)
    tracks = tracker.update([det_return])

    assert len(tracks) == 1
    assert tracks[0]["track_id"] == initial_id
    assert tracks[0]["state"] == TrackState.ACTIVE


# ===========================================================================
# SCENARIO 6: Face Returns & Identity Preserved
# ===========================================================================
def test_06_face_returns():
    stabilizer = TemporalStabilizer(unknown_persistence_duration=5)
    tid = 101

    # 1. Establish stable match
    res_match = RecognitionResult(
        face_id=str(tid),
        matched_student_id="STU_001",
        matched_student_name="Alice",
        similarity=0.88,
        status=RecognitionStatus.MATCH,
        threshold=0.65,
        embedding_model="w600k_mbf.onnx",
        processing_time_ms=10.0
    )
    for f in range(1, 4):
        sid, sname, sim, status = stabilizer.update_track_recognition(tid, f, res_match)
    assert sid == "STU_001"

    # 2. Face missing / drops for 2 frames
    # Stabilizer state remains intact
    assert stabilizer.get_track_summary(tid)["stable_student_id"] == "STU_001"

    # 3. Face returns, stabilizer immediately confirms Alice
    sid, sname, sim, status = stabilizer.update_track_recognition(tid, 6, res_match)
    assert sid == "STU_001"
    assert status == RecognitionStatus.MATCH


# ===========================================================================
# SCENARIO 7: Two Faces Cross Paths
# ===========================================================================
def test_07_two_faces_cross_paths():
    tracker = ByteTracker(match_thresh=0.7)

    # Face A moves right: x from 100 to 300
    # Face B moves left: x from 500 to 300
    id_a = None
    id_b = None

    for f in range(5):
        det_a = make_det(100 + f * 40, 200, 200 + f * 40, 320, conf=0.9)
        det_b = make_det(500 - f * 40, 200, 600 - f * 40, 320, conf=0.9)
        tracks = tracker.update([det_a, det_b])
        assert len(tracks) == 2
        tracks_by_x = sorted(tracks, key=lambda t: t["bbox"][0])
        if f == 0:
            id_a = tracks_by_x[0]["track_id"]
            id_b = tracks_by_x[1]["track_id"]
        else:
            # Face A is moving right, Face B is moving left
            assert tracks_by_x[0]["track_id"] == id_a
            assert tracks_by_x[1]["track_id"] == id_b

    assert id_a != id_b


# ===========================================================================
# SCENARIO 8: One Known Face + One Unknown Face
# ===========================================================================
def test_08_one_known_face_plus_one_unknown_face():
    stabilizer = TemporalStabilizer(min_stable_observations=3)
    tid_known = 1
    tid_unknown = 2

    res_known = RecognitionResult(
        face_id=str(tid_known),
        matched_student_id="STU_100",
        matched_student_name="Known Student",
        similarity=0.85,
        status=RecognitionStatus.MATCH,
        threshold=0.65,
        embedding_model="w600k_mbf.onnx",
        processing_time_ms=10.0
    )
    res_unknown = RecognitionResult(
        face_id=str(tid_unknown),
        matched_student_id=None,
        matched_student_name=None,
        similarity=0.45,
        status=RecognitionStatus.UNKNOWN,
        threshold=0.65,
        embedding_model="w600k_mbf.onnx",
        processing_time_ms=10.0
    )

    for f in range(1, 5):
        sid_k, _, _, stat_k = stabilizer.update_track_recognition(tid_known, f, res_known)
        sid_u, _, _, stat_u = stabilizer.update_track_recognition(tid_unknown, f, res_unknown)

    assert sid_k == "STU_100"
    assert stat_k == RecognitionStatus.MATCH

    assert sid_u is None
    assert stat_u == RecognitionStatus.UNKNOWN


# ===========================================================================
# SCENARIO 9: Multiple Known Faces
# ===========================================================================
def test_09_multiple_known_faces():
    stabilizer = TemporalStabilizer(min_stable_observations=2)
    students = [("STU_1", "Alice"), ("STU_2", "Bob"), ("STU_3", "Charlie")]

    for f in range(1, 4):
        for idx, (sid, name) in enumerate(students):
            tid = idx + 10
            res = RecognitionResult(
                face_id=str(tid),
                matched_student_id=sid,
                matched_student_name=name,
                similarity=0.82 + idx * 0.03,
                status=RecognitionStatus.MATCH,
                threshold=0.65,
                embedding_model="w600k_mbf.onnx",
                processing_time_ms=10.0
            )
            out_id, out_name, out_sim, out_stat = stabilizer.update_track_recognition(tid, f, res)

    for idx, (sid, name) in enumerate(students):
        tid = idx + 10
        summary = stabilizer.get_track_summary(tid)
        assert summary["stable_student_id"] == sid
        assert summary["stable_status"] == RecognitionStatus.MATCH


# ===========================================================================
# SCENARIO 10: Multiple Unknown Faces (Unknown Rejection Preserved)
# ===========================================================================
def test_10_multiple_unknown_faces():
    stabilizer = TemporalStabilizer()

    for tid in [50, 51, 52, 53]:
        for f in range(1, 6):
            res = RecognitionResult(
                face_id=str(tid),
                matched_student_id=None,
                matched_student_name=None,
                similarity=0.40,
                status=RecognitionStatus.UNKNOWN,
                threshold=0.65,
                embedding_model="w600k_mbf.onnx",
                processing_time_ms=10.0
            )
            out_id, _, _, out_stat = stabilizer.update_track_recognition(tid, f, res)
            assert out_id is None
            assert out_stat == RecognitionStatus.UNKNOWN

        summary = stabilizer.get_track_summary(tid)
        assert summary["stable_student_id"] is None
        assert summary["stable_status"] == RecognitionStatus.UNKNOWN


# ===========================================================================
# SCENARIO 11: Temporary Detection Failure
# ===========================================================================
def test_11_temporary_detection_failure():
    tracker = ByteTracker(max_lost_frames=5)
    det = make_det(200, 200, 320, 340, conf=0.92)

    # Stable for 4 frames
    for _ in range(4):
        tracker.update([det])
    tid = tracker.tracked_stracks[0].track_id

    # Frame 5: Detector drops face (0 detections)
    tracks_f5 = tracker.update([])
    assert len(tracks_f5) == 0

    # Frame 6: Detector recovers face
    tracks_f6 = tracker.update([det])
    assert len(tracks_f6) == 1
    assert tracks_f6[0]["track_id"] == tid
    assert tracks_f6[0]["state"] == TrackState.ACTIVE


# ===========================================================================
# SCENARIO 12: Temporary Recognition Failure (Flicker Suppression)
# ===========================================================================
def test_12_temporary_recognition_failure():
    stabilizer = TemporalStabilizer(min_stable_observations=3, unknown_persistence_duration=4)
    tid = 77

    res_match = RecognitionResult(
        face_id=str(tid),
        matched_student_id="STU_99",
        matched_student_name="Zoe",
        similarity=0.86,
        status=RecognitionStatus.MATCH,
        threshold=0.65,
        embedding_model="w600k_mbf.onnx",
        processing_time_ms=10.0
    )
    res_glitch = RecognitionResult(
        face_id=str(tid),
        matched_student_id=None,
        matched_student_name=None,
        similarity=0.52,
        status=RecognitionStatus.UNKNOWN,
        threshold=0.65,
        embedding_model="w600k_mbf.onnx",
        processing_time_ms=10.0
    )

    # Establish stable match (frames 1-3)
    for f in range(1, 4):
        stabilizer.update_track_recognition(tid, f, res_match)
    assert stabilizer.get_track_summary(tid)["stable_student_id"] == "STU_99"

    # Frame 4: Temporary recognition failure / drop to UNKNOWN
    sid_4, _, _, stat_4 = stabilizer.update_track_recognition(tid, 4, res_glitch)
    # Flicker suppression: identity MUST NOT DROP!
    assert sid_4 == "STU_99"
    assert stat_4 == RecognitionStatus.MATCH

    # Frame 5: Match recovers
    sid_5, _, _, stat_5 = stabilizer.update_track_recognition(tid, 5, res_match)
    assert sid_5 == "STU_99"
    assert stat_5 == RecognitionStatus.MATCH


# ===========================================================================
# SCENARIO 13: Poor-Quality Face Resilience
# ===========================================================================
def test_13_poor_quality_face_resilience():
    stabilizer = TemporalStabilizer(min_stable_observations=2, poor_quality_timeout=5)
    tid = 88

    res_good = RecognitionResult(
        face_id=str(tid),
        matched_student_id="STU_55",
        matched_student_name="David",
        similarity=0.84,
        status=RecognitionStatus.MATCH,
        threshold=0.65,
        embedding_model="w600k_mbf.onnx",
        processing_time_ms=10.0
    )
    qm_good = make_quality_result(str(tid), [100, 100, 200, 200], "RECOGNITION_READY", 90.0)
    qm_bad = make_quality_result(str(tid), [100, 100, 200, 200], "LOW_QUALITY", 25.0)

    # Confirm identity
    for f in range(1, 3):
        stabilizer.update_track_recognition(tid, f, res_good, quality_metrics=qm_good)
    assert stabilizer.get_track_summary(tid)["stable_student_id"] == "STU_55"

    # 3 frames of poor quality / motion blur
    for f in range(3, 6):
        sid, _, _, stat = stabilizer.update_track_recognition(tid, f, res_good, quality_metrics=qm_bad)
        # Identity protected
        assert sid == "STU_55"
        assert stat == RecognitionStatus.MATCH

    # When poor quality exceeds poor_quality_timeout (5 frames)
    for f in range(6, 9):
        sid, _, _, stat = stabilizer.update_track_recognition(tid, f, res_good, quality_metrics=qm_bad)
    # Must degrade to UNKNOWN after persistent poor quality
    assert sid is None
    assert stat == RecognitionStatus.UNKNOWN


# ===========================================================================
# SCENARIO 14: Identity Switch Attempt Protection
# ===========================================================================
def test_14_identity_switch_attempt_protection():
    stabilizer = TemporalStabilizer(min_stable_observations=3, identity_switch_threshold=4)
    tid = 99

    res_a = RecognitionResult(
        face_id=str(tid),
        matched_student_id="STU_A",
        matched_student_name="Alice",
        similarity=0.87,
        status=RecognitionStatus.MATCH,
        threshold=0.65,
        embedding_model="w600k_mbf.onnx",
        processing_time_ms=10.0
    )
    res_b = RecognitionResult(
        face_id=str(tid),
        matched_student_id="STU_B",
        matched_student_name="Bob",
        similarity=0.89,
        status=RecognitionStatus.MATCH,
        threshold=0.65,
        embedding_model="w600k_mbf.onnx",
        processing_time_ms=10.0
    )

    # Establish Student A
    for f in range(1, 4):
        stabilizer.update_track_recognition(tid, f, res_a)
    assert stabilizer.get_track_summary(tid)["stable_student_id"] == "STU_A"

    # Momentary noise / false recognition of Student B for 2 frames
    sid_4, _, _, _ = stabilizer.update_track_recognition(tid, 4, res_b)
    assert sid_4 == "STU_A"  # Protected!
    sid_5, _, _, _ = stabilizer.update_track_recognition(tid, 5, res_b)
    assert sid_5 == "STU_A"  # Still protected!

    # Return to A
    sid_6, _, _, _ = stabilizer.update_track_recognition(tid, 6, res_a)
    assert sid_6 == "STU_A"

    # Now sustained genuine transition to B for 4 consecutive frames (identity_switch_threshold)
    stabilizer.update_track_recognition(tid, 7, res_b)
    stabilizer.update_track_recognition(tid, 8, res_b)
    stabilizer.update_track_recognition(tid, 9, res_b)
    sid_10, _, _, stat_10 = stabilizer.update_track_recognition(tid, 10, res_b)

    # Only after 4 consecutive matches of B does it legitimately transition
    assert sid_10 == "STU_B"
    assert stat_10 == RecognitionStatus.MATCH


# ===========================================================================
# SCENARIO 15: Track Timeout and Removal
# ===========================================================================
def test_15_track_timeout_and_removal():
    tracker = ByteTracker(max_lost_frames=4)
    det = make_det(100, 100, 200, 200, conf=0.9)

    tracker.update([det])
    tid = tracker.tracked_stracks[0].track_id

    # Face disappears
    for f in range(1, 6):
        tracker.update([])

    # Must be in removed_stracks, not in tracked or lost
    assert not any(t.track_id == tid for t in tracker.tracked_stracks)
    assert not any(t.track_id == tid for t in tracker.lost_stracks)
    assert any(t.track_id == tid for t in tracker.removed_stracks)


# ===========================================================================
# SCENARIO 16: Track Memory Cleanup
# ===========================================================================
def test_16_track_cleanup_memory():
    stabilizer = TemporalStabilizer()
    for tid in range(1, 11):
        res = RecognitionResult(
            face_id=str(tid),
            matched_student_id=f"STU_{tid}",
            matched_student_name=f"Student {tid}",
            similarity=0.85,
            status=RecognitionStatus.MATCH,
            threshold=0.65,
            embedding_model="w600k_mbf.onnx",
            processing_time_ms=10.0
        )
        stabilizer.update_track_recognition(tid, 1, res)

    assert len(stabilizer.track_states) == 10
    assert len(stabilizer.track_histories) == 10

    # Retain only tracks 1, 2, 3
    stabilizer.cleanup_tracks(active_track_ids={1, 2, 3})
    assert len(stabilizer.track_states) == 3
    assert len(stabilizer.track_histories) == 3
    assert set(stabilizer.track_states.keys()) == {1, 2, 3}


# ===========================================================================
# SCENARIO 17: Simultaneous Independent Tracks
# ===========================================================================
def test_17_simultaneous_independent_tracks():
    tracker = ByteTracker()
    n_faces = 20

    for f in range(10):
        dets = [
            make_det(i * 60, 100 + (f % 3) * 2, i * 60 + 50, 160 + (f % 3) * 2, conf=0.91)
            for i in range(n_faces)
        ]
        tracks = tracker.update(dets)
        assert len(tracks) == n_faces

    # Verify all 20 have unique track IDs
    tids = [t["track_id"] for t in tracks]
    assert len(set(tids)) == n_faces


# ===========================================================================
# SCENARIO 18: Long-Running Tracking Stress
# ===========================================================================
def test_18_long_running_tracking_stress():
    tracker = ByteTracker(max_lost_frames=10)
    stabilizer = TemporalStabilizer()

    # 100 frames simulation with 5 continuous tracks + dynamic perturbations
    for f in range(1, 101):
        dets = []
        for i in range(5):
            # Smooth circular motion
            theta = (f + i * 20) * 0.1
            cx = 300 + i * 200 + int(np.cos(theta) * 15)
            cy = 300 + int(np.sin(theta) * 15)
            dets.append(make_det(cx - 40, cy - 50, cx + 40, cy + 50, conf=0.92))

        tracks = tracker.update(dets)
        assert len(tracks) == 5

        # Update stabilizer
        for t in tracks:
            tid = t["track_id"]
            res = RecognitionResult(
                face_id=str(tid),
                matched_student_id=f"STU_{tid % 3}",
                matched_student_name=f"Student {tid % 3}",
                similarity=0.88,
                status=RecognitionStatus.MATCH,
                threshold=0.65,
                embedding_model="w600k_mbf.onnx",
                processing_time_ms=5.0
            )
            stabilizer.update_track_recognition(tid, f, res)

    # Memory check: History deque must stay bounded at maxlen
    for tid, history in stabilizer.track_histories.items():
        assert len(history) <= stabilizer.history_length
