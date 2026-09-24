import os
import sys
import time
import cv2
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.face_quality import FaceQualityAssessor
from core.face_alignment import FaceAligner
from core.face_processor import FaceProcessor
from core.schemas import FaceQualityResult, RecognitionReadyFace
from config.settings import get_settings


def make_test_face(w: int, h: int, base_color: tuple = (180, 180, 180), blur: bool = False) -> np.ndarray:
    """Helper to generate a textured synthetic face with discernible facial features."""
    face = np.full((h, w, 3), base_color, dtype=np.uint8)
    b_val = float(np.mean(base_color))
    feat_color = (0, 0, 0) if b_val > 40 else (5, 5, 5)

    eye_radius = max(2, int(min(w, h) * 0.06))
    left_eye = (int(w * 0.35), int(h * 0.35))
    right_eye = (int(w * 0.65), int(h * 0.35))
    cv2.circle(face, left_eye, eye_radius, feat_color, -1)
    cv2.circle(face, right_eye, eye_radius, feat_color, -1)

    cv2.line(face, (int(w * 0.5), int(h * 0.45)), (int(w * 0.5), int(h * 0.6)), feat_color, max(1, int(w * 0.03)))
    cv2.rectangle(face, (int(w * 0.35), int(h * 0.75)), (int(w * 0.65), int(h * 0.82)), feat_color, -1)

    if blur:
        ksize = max(15, (w // 8) * 2 + 1)
        face = cv2.GaussianBlur(face, (ksize, ksize), 0)
    return face


# ---------------------------------------------------------------------------
# Requirement 1: Face Quality Assessment
# ---------------------------------------------------------------------------
def test_req1_face_quality_assessment():
    assessor = FaceQualityAssessor()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    face_img = make_test_face(200, 250, (180, 180, 180), blur=False)
    frame[100:350, 100:300] = face_img

    det = {
        "bbox": [100, 100, 300, 350],
        "face_area": 50000,
        "keypoints": [[135, 170], [165, 170], [150, 200], [135, 250], [165, 250]]
    }
    result = assessor.assess(det, frame, "face_req1")

    assert isinstance(result, FaceQualityResult)
    assert result.face_id == "face_req1"
    assert result.quality_status == "RECOGNITION_READY"
    assert result.rejection_reason is None
    assert result.area == 50000
    assert result.sharpness >= 50.0
    assert 30.0 <= result.brightness <= 230.0


# ---------------------------------------------------------------------------
# Requirement 2: Face Size Measurement
# ---------------------------------------------------------------------------
def test_req2_face_size_measurement():
    assessor = FaceQualityAssessor()
    settings = get_settings().quality
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Sub-test 2a: Small face below min_face_area threshold (e.g. 1200 < 3000)
    small_face = make_test_face(30, 40, (180, 180, 180), blur=False)
    frame[10:50, 10:40] = small_face
    small_det = {"bbox": [10, 10, 40, 50], "face_area": 1200}
    res_small = assessor.assess(small_det, frame, "small_face")
    assert res_small.width == 30
    assert res_small.height == 40
    assert res_small.area == 1200
    assert res_small.quality_status == "LOW_QUALITY"
    assert "Too small" in res_small.rejection_reason

    # Sub-test 2b: Standard face above min_face_area (e.g. 20000 >= 3000)
    norm_face = make_test_face(100, 200, (180, 180, 180), blur=False)
    frame[200:400, 200:300] = norm_face
    norm_det = {"bbox": [200, 200, 300, 400], "face_area": 20000}
    res_norm = assessor.assess(norm_det, frame, "norm_face")
    assert res_norm.area >= settings.min_face_area
    assert res_norm.quality_status == "RECOGNITION_READY"


# ---------------------------------------------------------------------------
# Requirement 3: Blur / Sharpness Measurement
# ---------------------------------------------------------------------------
def test_req3_blur_sharpness_measurement():
    assessor = FaceQualityAssessor()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Sharp face
    sharp_crop = make_test_face(150, 150, (180, 180, 180), blur=False)
    sharp_val = assessor.evaluate_sharpness(sharp_crop)
    assert sharp_val > 50.0

    # Blurry face
    blurry_crop = make_test_face(150, 150, (180, 180, 180), blur=True)
    blurry_val = assessor.evaluate_sharpness(blurry_crop)
    assert blurry_val < 50.0

    frame[100:250, 100:250] = blurry_crop
    det = {"bbox": [100, 100, 250, 250], "face_area": 22500}
    res = assessor.assess(det, frame, "blurry_face")
    assert res.quality_status == "LOW_QUALITY"
    assert "Too blurry" in res.rejection_reason


# ---------------------------------------------------------------------------
# Requirement 4: Brightness Measurement
# ---------------------------------------------------------------------------
def test_req4_brightness_measurement():
    assessor = FaceQualityAssessor()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Sub-test 4a: Dark Face (< 30)
    dark_crop = make_test_face(150, 150, (15, 15, 15), blur=False)
    frame[50:200, 50:200] = dark_crop
    dark_det = {"bbox": [50, 50, 200, 200], "face_area": 22500}
    res_dark = assessor.assess(dark_det, frame, "dark_face")
    assert res_dark.brightness < 30.0
    assert res_dark.quality_status == "LOW_QUALITY"
    assert "Too dark" in res_dark.rejection_reason

    # Sub-test 4b: Glare / Overexposed Face (> 230)
    bright_crop = make_test_face(150, 150, (250, 250, 250), blur=False)
    frame[300:450, 300:450] = bright_crop
    bright_det = {"bbox": [300, 300, 450, 450], "face_area": 22500}
    res_bright = assessor.assess(bright_det, frame, "bright_face")
    assert res_bright.brightness > 230.0
    assert res_bright.quality_status == "LOW_QUALITY"
    assert "Too bright" in res_bright.rejection_reason

    # Sub-test 4c: Normal Brightness (between 30 and 230)
    normal_crop = make_test_face(150, 150, (140, 140, 140), blur=False)
    frame[600:750, 600:750] = normal_crop
    norm_det = {"bbox": [600, 600, 750, 750], "face_area": 22500}
    res_normal = assessor.assess(norm_det, frame, "norm_bright")
    assert 30.0 <= res_normal.brightness <= 230.0
    assert res_normal.quality_status == "RECOGNITION_READY"


# ---------------------------------------------------------------------------
# Requirement 5: Face Crop
# ---------------------------------------------------------------------------
def test_req5_face_crop_safety_and_boundaries():
    assessor = FaceQualityAssessor()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Sub-test 5a: Face crossing right/bottom frame boundary
    boundary_face = make_test_face(100, 100, (180, 180, 180), blur=False)
    frame[1000:1080, 1850:1920] = boundary_face[:80, :70]
    det_boundary = {"bbox": [1850, 1000, 1950, 1100], "face_area": 10000}
    res = assessor.assess(det_boundary, frame, "boundary_face")
    assert res.bbox == [1850, 1000, 1920, 1080]
    assert res.quality_status == "RECOGNITION_READY"

    # Sub-test 5b: Completely out-of-bounds coordinates
    det_oob = {"bbox": [2000, 2000, 2100, 2100], "face_area": 10000}
    res_oob = assessor.assess(det_oob, frame, "oob_face")
    assert res_oob.quality_status == "LOW_QUALITY"
    assert "outside frame boundaries" in res_oob.rejection_reason


# ---------------------------------------------------------------------------
# Requirement 6: Face Alignment
# ---------------------------------------------------------------------------
def test_req6_face_alignment():
    aligner = FaceAligner(output_size=(112, 112))
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    # Render face with keypoints
    face_crop = make_test_face(200, 200, (180, 180, 180))
    frame[100:300, 200:400] = face_crop

    # 5 standard facial keypoints
    keypoints = [
        [270.0, 170.0],  # Left eye
        [330.0, 170.0],  # Right eye
        [300.0, 200.0],  # Nose
        [275.0, 240.0],  # Left mouth
        [325.0, 240.0],  # Right mouth
    ]

    aligned = aligner.align(frame, keypoints)
    assert isinstance(aligned, np.ndarray)
    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == np.uint8


# ---------------------------------------------------------------------------
# Requirement 7: Normalization / Preprocessing
# ---------------------------------------------------------------------------
def test_req7_normalization_preprocessing():
    aligner = FaceAligner()
    # Test with known BGR values
    aligned_bgr = np.full((112, 112, 3), (255, 128, 0), dtype=np.uint8)

    tensor = aligner.normalize(aligned_bgr)
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 3, 112, 112)
    assert tensor.dtype == np.float32

    # Check ArcFace normalization: (val - 127.5) / 128.0
    # RGB conversion swaps B (ch 0) and R (ch 2):
    # In aligned_bgr, B=255 (ch 0), G=128 (ch 1), R=0 (ch 2)
    # After cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB):
    # Output channel 0 is R (orig ch 2 = 0) -> (0 - 127.5)/128 = ~ -0.9961
    # Output channel 1 is G (orig ch 1 = 128) -> (128 - 127.5)/128 = ~0.0039
    # Output channel 2 is B (orig ch 0 = 255) -> (255 - 127.5)/128 = ~0.9961
    r_val = tensor[0, 0, 0, 0]
    g_val = tensor[0, 1, 0, 0]
    b_val = tensor[0, 2, 0, 0]

    assert abs(r_val - ((0 - 127.5) / 128.0)) < 1e-4
    assert abs(g_val - ((128 - 127.5) / 128.0)) < 1e-4
    assert abs(b_val - ((255 - 127.5) / 128.0)) < 1e-4

    # Verify tensor bounds [-1.0, 1.0]
    assert np.all(tensor >= -1.0)
    assert np.all(tensor <= 1.0)


# ---------------------------------------------------------------------------
# Requirement 8: Multiple-Face Processing
# ---------------------------------------------------------------------------
def test_req8_multiple_face_processing():
    processor = FaceProcessor()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # 4 synthetic faces in the frame
    # Face 1: High quality
    frame[50:250, 50:250] = make_test_face(200, 200, (180, 180, 180), blur=False)
    # Face 2: Blurry
    frame[50:250, 350:550] = make_test_face(200, 200, (180, 180, 180), blur=True)
    # Face 3: Too small (< 3000 px)
    frame[50:90, 650:680] = make_test_face(30, 40, (180, 180, 180), blur=False)
    # Face 4: High quality
    frame[50:250, 800:1000] = make_test_face(200, 200, (180, 180, 180), blur=False)

    detections = [
        {
            "track_id": "stud_1",
            "bbox": [50, 50, 250, 250],
            "face_area": 40000,
            "keypoints": [[120, 120], [180, 120], [150, 150], [125, 190], [175, 190]],
        },
        {
            "track_id": "stud_2",
            "bbox": [350, 50, 550, 250],
            "face_area": 40000,
            "keypoints": [[420, 120], [480, 120], [450, 150], [425, 190], [475, 190]],
        },
        {
            "track_id": "stud_3",
            "bbox": [650, 50, 680, 90],
            "face_area": 1200,
            "keypoints": [[660, 60], [670, 60], [665, 70], [662, 80], [668, 80]],
        },
        {
            "track_id": "stud_4",
            "bbox": [800, 50, 1000, 250],
            "face_area": 40000,
            "keypoints": [[870, 120], [930, 120], [900, 150], [875, 190], [925, 190]],
        },
    ]

    ready_faces, rejected_faces = processor.process(detections, frame)

    assert len(ready_faces) == 2
    assert len(rejected_faces) == 2

    ready_ids = [r.quality_metrics.face_id for r in ready_faces]
    assert "stud_1" in ready_ids
    assert "stud_4" in ready_ids

    rejected_ids = [r.face_id for r in rejected_faces]
    assert "stud_2" in rejected_ids  # Blurry
    assert "stud_3" in rejected_ids  # Too small


# ---------------------------------------------------------------------------
# Requirement 9: Near / Middle / Far Face Testing
# ---------------------------------------------------------------------------
def test_req9_near_middle_far_zones():
    assessor = FaceQualityAssessor()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # NEAR Zone: Area > 15,000 px (e.g. 150x150 = 22,500)
    frame[50:200, 50:200] = make_test_face(150, 150, (180, 180, 180), blur=False)
    det_near = {"bbox": [50, 50, 200, 200], "face_area": 22500, "zone": "NEAR"}
    res_near = assessor.assess(det_near, frame, "near_face")
    assert res_near.quality_status == "RECOGNITION_READY"

    # MIDDLE Zone: Area 3,000 to 15,000 px (e.g. 80x100 = 8,000)
    frame[300:400, 300:380] = make_test_face(80, 100, (180, 180, 180), blur=False)
    det_middle = {"bbox": [300, 300, 380, 400], "face_area": 8000, "zone": "MIDDLE"}
    res_middle = assessor.assess(det_middle, frame, "middle_face")
    assert res_middle.quality_status == "RECOGNITION_READY"

    # FAR Zone: Area < 3,000 px (e.g. 40x50 = 2,000)
    frame[500:550, 500:540] = make_test_face(40, 50, (180, 180, 180), blur=False)
    det_far = {"bbox": [500, 500, 540, 550], "face_area": 2000, "zone": "FAR"}
    res_far = assessor.assess(det_far, frame, "far_face")
    assert res_far.quality_status == "LOW_QUALITY"
    assert "Too small" in res_far.rejection_reason


# ---------------------------------------------------------------------------
# Requirement 10: Recognition-Ready Output
# ---------------------------------------------------------------------------
def test_req10_recognition_ready_output_schema():
    processor = FaceProcessor()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[100:300, 100:300] = make_test_face(200, 200, (180, 180, 180))

    det = {
        "track_id": "stud_101",
        "bbox": [100, 100, 300, 300],
        "face_area": 40000,
        "keypoints": [[170, 170], [230, 170], [200, 200], [175, 240], [225, 240]],
    }
    fixed_ts = 1700000000.5
    ready_faces, rejected_faces = processor.process([det], frame, timestamp=fixed_ts)

    assert len(ready_faces) == 1
    rf = ready_faces[0]

    # Validate all fields in RecognitionReadyFace
    assert isinstance(rf, RecognitionReadyFace)
    assert rf.timestamp == fixed_ts
    assert rf.original_bbox == [100, 100, 300, 300]
    assert rf.aligned_face_tensor.shape == (1, 3, 112, 112)
    assert rf.aligned_face_tensor.dtype == np.float32

    # Validate nested FaceQualityResult
    qm = rf.quality_metrics
    assert isinstance(qm, FaceQualityResult)
    assert qm.face_id == "stud_101"
    assert qm.quality_status == "RECOGNITION_READY"
    assert qm.area == 40000
    assert qm.sharpness > 50.0
    assert 30.0 <= qm.brightness <= 230.0


# ---------------------------------------------------------------------------
# Requirement 11: Error Handling
# ---------------------------------------------------------------------------
def test_req11_error_handling():
    assessor = FaceQualityAssessor()
    aligner = FaceAligner()
    processor = FaceProcessor()

    # 11a: None frame passed to assessor
    res_none = assessor.assess({"bbox": [0, 0, 10, 10], "face_area": 100}, None, "err_1")
    assert res_none.quality_status == "LOW_QUALITY"
    assert "Invalid or empty image frame" in res_none.rejection_reason

    # 11b: Empty frame passed to assessor
    empty_frame = np.array([], dtype=np.uint8)
    res_empty = assessor.assess({"bbox": [0, 0, 10, 10], "face_area": 100}, empty_frame, "err_2")
    assert res_empty.quality_status == "LOW_QUALITY"

    # 11c: Aligner invalid inputs
    with pytest.raises(ValueError):
        aligner.align(None, [[0, 0]] * 5)

    with pytest.raises(ValueError):
        # Only 4 keypoints
        aligner.align(np.zeros((100, 100, 3), dtype=np.uint8), [[0, 0]] * 4)

    with pytest.raises(ValueError):
        # Empty keypoints
        aligner.align(np.zeros((100, 100, 3), dtype=np.uint8), [])

    # 11d: Normalize invalid inputs
    with pytest.raises(ValueError):
        aligner.normalize(None)

    with pytest.raises(ValueError):
        aligner.normalize(np.array([], dtype=np.uint8))

    # 11e: Processor handles None frame
    ready, rejected = processor.process([{"bbox": [10, 10, 50, 50]}], None)
    assert ready == []
    assert rejected == []


# ---------------------------------------------------------------------------
# Requirement 12: Performance / FPS / Latency
# ---------------------------------------------------------------------------
def test_req12_performance_latency_fps():
    processor = FaceProcessor()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Populate 10 faces in frame
    detections = []
    for i in range(10):
        x1 = 50 + (i % 5) * 200
        y1 = 50 + (i // 5) * 250
        x2, y2 = x1 + 150, y1 + 180
        frame[y1:y2, x1:x2] = make_test_face(150, 180, (180, 180, 180))
        detections.append({
            "track_id": f"perf_{i}",
            "bbox": [x1, y1, x2, y2],
            "face_area": 150 * 180,
            "keypoints": [
                [x1 + 40, y1 + 50], [x1 + 110, y1 + 50],
                [x1 + 75, y1 + 90],
                [x1 + 45, y1 + 130], [x1 + 105, y1 + 130]
            ]
        })

    # Warmup
    processor.process(detections[:1], frame)

    # Benchmark 20 iterations across 10 faces
    iterations = 20
    t0 = time.perf_counter()
    for _ in range(iterations):
        ready, _ = processor.process(detections, frame)
    t1 = time.perf_counter()

    total_time = t1 - t0
    total_faces = iterations * len(detections)
    latency_per_face_ms = (total_time / total_faces) * 1000.0
    throughput_faces_per_sec = total_faces / total_time
    fps_at_10_faces = iterations / total_time

    # Assert real-time processing performance:
    # Less than 10ms per face on CPU (typically ~0.5ms - 1.5ms)
    assert latency_per_face_ms < 10.0
    # Must process at least 100 faces/sec
    assert throughput_faces_per_sec > 100.0
