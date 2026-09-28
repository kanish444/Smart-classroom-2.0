import os
import sys
import time
import cv2
import numpy as np
import pytest

from core.scrfd_detector import SCRFDDetector
from core.tracking_pipeline import TrackingPipeline
from core.recognizer import FaceRecognizer
from core.face_embedder import ArcFaceEmbedder
from core.vector_store import FaissVectorStore
from core.schemas import RecognitionStatus, TrackedFace
from config.settings import get_settings


@pytest.fixture(scope="module")
def scrfd_detector():
    """Module-level SCRFD detector fixture."""
    detector = SCRFDDetector(conf_thresh=0.35, nms_thresh=0.40)
    return detector


@pytest.fixture(scope="module")
def base_face_crop():
    """Generates a textured synthetic face with discernible eyes, nose, and mouth."""
    w, h = 180, 180
    face = np.full((h, w, 3), (180, 180, 180), dtype=np.uint8)
    feat_color = (15, 15, 15)

    # Eyes
    cv2.circle(face, (int(w * 0.32), int(h * 0.35)), int(w * 0.08), feat_color, -1)
    cv2.circle(face, (int(w * 0.68), int(h * 0.35)), int(w * 0.08), feat_color, -1)
    cv2.circle(face, (int(w * 0.32), int(h * 0.35)), int(w * 0.03), (255, 255, 255), -1)
    cv2.circle(face, (int(w * 0.68), int(h * 0.35)), int(w * 0.03), (255, 255, 255), -1)

    # Nose
    cv2.line(face, (int(w * 0.50), int(h * 0.44)), (int(w * 0.50), int(h * 0.60)), feat_color, 2)
    cv2.line(face, (int(w * 0.45), int(h * 0.60)), (int(w * 0.55), int(h * 0.60)), feat_color, 2)

    # Mouth
    cv2.rectangle(face, (int(w * 0.34), int(h * 0.74)), (int(w * 0.66), int(h * 0.82)), feat_color, -1)

    # If genuine student photo sample exists, use real face crop
    sample_path = "data/enrollment/922524243069/sample_01.jpg"
    if os.path.exists(sample_path):
        sample = cv2.imread(sample_path)
        if sample is not None and sample.shape[0] > 200:
            # Genuine crop [419, 193, 654, 480]
            real_crop = sample[193:480, 419:654]
            return real_crop

    return face


# ---------------------------------------------------------------------------
# TEST 1: One Large Face
# ---------------------------------------------------------------------------
def test_01_one_large_face(scrfd_detector, base_face_crop):
    frame = np.full((720, 1280, 3), (200, 205, 210), dtype=np.uint8)
    crop = cv2.resize(base_face_crop, (200, 200))
    frame[260:460, 540:740] = crop

    dets = scrfd_detector.detect(frame)
    assert len(dets) == 1, f"Expected exactly 1 detection, got {len(dets)}"
    det = dets[0]
    x1, y1, x2, y2 = det["bbox"]
    assert 500 <= x1 <= 560
    assert 240 <= y1 <= 280
    assert 720 <= x2 <= 780
    assert 440 <= y2 <= 480
    assert det["confidence"] >= 0.50
    assert len(det["keypoints"]) == 5


# ---------------------------------------------------------------------------
# TEST 2: Three Large Faces
# ---------------------------------------------------------------------------
def test_02_three_large_faces(scrfd_detector, base_face_crop):
    frame = np.full((720, 1280, 3), (200, 205, 210), dtype=np.uint8)
    crop = cv2.resize(base_face_crop, (150, 150))

    # Place 3 separate faces
    frame[200:350, 150:300] = crop
    frame[200:350, 565:715] = crop
    frame[200:350, 980:1130] = crop

    dets = scrfd_detector.detect(frame)
    assert len(dets) == 3, f"Expected 3 detections, got {len(dets)}"
    x_centers = sorted([(d["bbox"][0] + d["bbox"][2]) // 2 for d in dets])
    assert 200 <= x_centers[0] <= 250
    assert 615 <= x_centers[1] <= 665
    assert 1030 <= x_centers[2] <= 1080


# ---------------------------------------------------------------------------
# TEST 3: 10+ Faces (Dense Grid)
# ---------------------------------------------------------------------------
def test_03_ten_plus_faces(scrfd_detector, base_face_crop):
    frame = np.full((720, 1280, 3), (190, 195, 200), dtype=np.uint8)
    crop = cv2.resize(base_face_crop, (80, 80))

    # Place 12 faces in 3 rows x 4 cols
    for row in range(3):
        y = 120 + row * 180
        for col in range(4):
            x = 100 + col * 280
            frame[y:y+80, x:x+80] = crop

    dets = scrfd_detector.detect(frame)
    assert len(dets) >= 10, f"Expected at least 10 independent detections, got {len(dets)}"


# ---------------------------------------------------------------------------
# TEST 4: Crowded Classroom Image
# ---------------------------------------------------------------------------
def test_04_crowded_classroom_detection(scrfd_detector, base_face_crop):
    frame = np.full((720, 1280, 3), (190, 195, 200), dtype=np.uint8)
    
    # 5 Front row large faces (120x120)
    front_crop = cv2.resize(base_face_crop, (120, 120))
    for i in range(5):
        frame[450:570, 80 + i * 230 : 200 + i * 230] = front_crop

    # 5 Middle row faces (70x70)
    mid_crop = cv2.resize(base_face_crop, (70, 70))
    for i in range(5):
        frame[280:350, 120 + i * 220 : 190 + i * 220] = mid_crop

    # 4 Back row small faces (45x45)
    back_crop = cv2.resize(base_face_crop, (45, 45))
    for i in range(4):
        frame[160:205, 200 + i * 240 : 245 + i * 240] = back_crop

    total_faces_placed = 14
    dets = scrfd_detector.detect(frame)
    recall = len(dets) / total_faces_placed
    assert recall >= 0.70, f"Expected recall >= 70%, got {recall:.2f} ({len(dets)}/{total_faces_placed})"


# ---------------------------------------------------------------------------
# TEST 5: Small Distant Faces
# ---------------------------------------------------------------------------
def test_05_small_distant_faces(scrfd_detector, base_face_crop):
    frame = np.full((720, 1280, 3), (190, 195, 200), dtype=np.uint8)

    # 40x40 distant student
    crop_40 = cv2.resize(base_face_crop, (40, 40))
    frame[180:220, 500:540] = crop_40

    # Multi-scale / tiled detection for small face recovery (Section 7)
    dets = scrfd_detector.detect(frame, run_tiles=True)
    assert len(dets) >= 1, "Expected small distant face to be detected by SCRFD"
    small_det = dets[0]
    assert small_det["zone"] == "FAR"
    assert small_det["face_area"] <= 3000


# ---------------------------------------------------------------------------
# TEST 6: Partially Visible Face (Edge of Frame)
# ---------------------------------------------------------------------------
def test_06_partially_visible_face(scrfd_detector, base_face_crop):
    frame = np.full((720, 1280, 3), (190, 195, 200), dtype=np.uint8)
    crop = cv2.resize(base_face_crop, (80, 80))

    # Face clipped at right edge: only left 35px is visible
    frame[300:380, 1245:1280] = crop[:, :35]

    dets = scrfd_detector.detect(frame)
    if len(dets) > 0:
        det = dets[0]
        bx1, by1, bx2, by2 = det["bbox"]
        assert bx2 <= 1280, "Bounding box must not exceed frame boundary"
        aspect_ratio = (bx2 - bx1) / float(max(1, by2 - by1))
        # Should have truncated aspect ratio or small area
        assert aspect_ratio < 0.65 or (bx2 - bx1) <= 50


# ---------------------------------------------------------------------------
# TEST 7: Back of Head (Rejection)
# ---------------------------------------------------------------------------
def test_07_back_of_head_rejection(scrfd_detector):
    frame = np.full((720, 1280, 3), (200, 205, 210), dtype=np.uint8)
    
    # Back of head patch (hair texture, no eyes/nose/mouth)
    head = np.full((120, 120, 3), (25, 20, 15), dtype=np.uint8)
    cv2.circle(head, (60, 60), 55, (10, 8, 5), -1)
    frame[300:420, 580:700] = head

    dets = scrfd_detector.detect(frame)
    assert len(dets) == 0, f"Expected 0 detections for back of head, got {len(dets)}"


# ---------------------------------------------------------------------------
# TEST 8: Unknown Person
# ---------------------------------------------------------------------------
def test_08_unknown_person(scrfd_detector, base_face_crop):
    pipeline = TrackingPipeline(detector=scrfd_detector)
    frame = np.full((720, 1280, 3), (200, 205, 210), dtype=np.uint8)
    
    # Unenrolled synthetic face
    stranger = np.full((150, 150, 3), (120, 130, 140), dtype=np.uint8)
    cv2.circle(stranger, (45, 50), 10, (10, 10, 10), -1)
    cv2.circle(stranger, (105, 50), 10, (10, 10, 10), -1)
    cv2.line(stranger, (75, 65), (75, 90), (10, 10, 10), 2)
    cv2.rectangle(stranger, (50, 110), (100, 125), (10, 10, 10), -1)
    frame[250:400, 500:650] = stranger

    # Process 3 frames
    for _ in range(3):
        tracks = pipeline.process_frame(frame)

    if len(tracks) > 0:
        track = tracks[0]
        assert track.stable_student_id is None, "Stranger must not be matched to enrolled student"
        assert track.display_status in ("UNKNOWN", "VERIFYING", "LOW QUALITY", "FACE TOO SMALL")


# ---------------------------------------------------------------------------
# TEST 9: Same Student Across Multiple Frames (ByteTrack Stability)
# ---------------------------------------------------------------------------
def test_09_same_student_across_multiple_frames(scrfd_detector, base_face_crop):
    pipeline = TrackingPipeline(detector=scrfd_detector)
    crop = cv2.resize(base_face_crop, (150, 150))

    initial_track_id = None
    for frame_idx in range(5):
        frame = np.full((720, 1280, 3), (200, 205, 210), dtype=np.uint8)
        # Shift slightly (2px per frame) to simulate subtle natural motion
        shift_x = frame_idx * 2
        frame[250:400, 500 + shift_x : 650 + shift_x] = crop

        tracks = pipeline.process_frame(frame)
        assert len(tracks) == 1, f"Expected 1 track at frame {frame_idx}, got {len(tracks)}"
        if initial_track_id is None:
            initial_track_id = tracks[0].track_id
        else:
            assert tracks[0].track_id == initial_track_id, (
                f"Track ID changed from {initial_track_id} to {tracks[0].track_id}"
            )


# ---------------------------------------------------------------------------
# TEST 10: Two Nearby Faces
# ---------------------------------------------------------------------------
def test_10_two_nearby_faces(scrfd_detector, base_face_crop):
    frame = np.full((720, 1280, 3), (200, 205, 210), dtype=np.uint8)
    crop = cv2.resize(base_face_crop, (120, 120))

    # Face 1: at x=400 (w=120, ends at x=520)
    # Face 2: at x=545 (separated by only 25 pixels)
    frame[250:370, 400:520] = crop
    frame[250:370, 545:665] = crop

    dets = scrfd_detector.detect(frame)
    assert len(dets) == 2, f"Expected 2 separate detections for nearby faces, got {len(dets)}"


# ---------------------------------------------------------------------------
# TEST 11: DroidCam / Latest-Frame Buffer Check
# ---------------------------------------------------------------------------
def test_11_latest_frame_buffer_droidcam():
    from camera.smartboard_camera import SmartBoardCamera
    cam = SmartBoardCamera()
    try:
        # Verify single-slot thread-safe buffer and lock
        assert hasattr(cam, "_latest_frame")
        assert hasattr(cam, "_frame_lock")
        assert hasattr(cam, "_lock")
    finally:
        cam.release()


# ---------------------------------------------------------------------------
# TEST 12: 1280x720 Coordinate Mapping & Overlay
# ---------------------------------------------------------------------------
def test_12_1280x720_coordinate_mapping_and_overlay(scrfd_detector, base_face_crop):
    pipeline = TrackingPipeline(detector=scrfd_detector)
    frame = np.full((720, 1280, 3), (190, 195, 200), dtype=np.uint8)
    crop = cv2.resize(base_face_crop, (160, 160))
    frame[300:460, 800:960] = crop

    tracks = pipeline.process_frame(frame)
    assert len(tracks) >= 1

    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        assert 0 <= x1 < 1280
        assert 0 <= y1 < 720
        assert 0 < x2 <= 1280
        assert 0 < y2 <= 720

    # Draw HUD overlay
    vis = pipeline.draw_debug_overlay(frame, tracks, fps=28.5)
    assert vis.shape == (720, 1280, 3)
    assert vis.dtype == np.uint8
