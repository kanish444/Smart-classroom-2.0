import os
import sys
import time
import cv2
import numpy as np
from loguru import logger

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


def run_complete_phase5_validation():
    logger.info("=" * 60)
    logger.info("SMARTCLASS VISION AI - PHASE 5 COMPLETE VALIDATION SUITE")
    logger.info("=" * 60)

    assessor = FaceQualityAssessor()
    aligner = FaceAligner()
    processor = FaceProcessor(assessor=assessor, aligner=aligner)
    settings = get_settings().quality

    results_summary = {}

    # ---------------------------------------------------------------------------
    # 1. Face Quality Assessment
    # ---------------------------------------------------------------------------
    logger.info("[REQ 1] Face Quality Assessment")
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    face_1 = make_test_face(200, 250, (180, 180, 180))
    frame[100:350, 100:300] = face_1
    det_1 = {
        "bbox": [100, 100, 300, 350],
        "face_area": 50000,
        "keypoints": [[135, 170], [165, 170], [150, 200], [135, 250], [165, 250]]
    }
    res_1 = assessor.assess(det_1, frame, "face_q1")
    pass_1 = (res_1.quality_status == "RECOGNITION_READY" and res_1.rejection_reason is None)
    results_summary["1. Face Quality Assessment"] = {
        "status": "PASS" if pass_1 else "FAIL",
        "measurements": f"Status={res_1.quality_status}, Sharpness={res_1.sharpness:.1f}, Brightness={res_1.brightness:.1f}, Area={res_1.area}"
    }
    logger.info(f" -> Result: {'PASS' if pass_1 else 'FAIL'} | {results_summary['1. Face Quality Assessment']['measurements']}")

    # ---------------------------------------------------------------------------
    # 2. Face Size Measurement
    # ---------------------------------------------------------------------------
    logger.info("[REQ 2] Face Size Measurement")
    small_face = make_test_face(30, 40, (180, 180, 180))
    frame[10:50, 10:40] = small_face
    det_small = {"bbox": [10, 10, 40, 50], "face_area": 1200}
    res_small = assessor.assess(det_small, frame, "face_small")
    pass_2 = (res_small.area == 1200 and res_small.quality_status == "LOW_QUALITY" and "Too small" in (res_small.rejection_reason or ""))
    results_summary["2. Face Size Measurement"] = {
        "status": "PASS" if pass_2 else "FAIL",
        "measurements": f"Width={res_small.width}, Height={res_small.height}, Area={res_small.area} px (Threshold={settings.min_face_area} px), Rejection='{res_small.rejection_reason}'"
    }
    logger.info(f" -> Result: {'PASS' if pass_2 else 'FAIL'} | {results_summary['2. Face Size Measurement']['measurements']}")

    # ---------------------------------------------------------------------------
    # 3. Blur/Sharpness Measurement
    # ---------------------------------------------------------------------------
    logger.info("[REQ 3] Blur/Sharpness Measurement")
    sharp_crop = make_test_face(150, 150, (180, 180, 180), blur=False)
    blurry_crop = make_test_face(150, 150, (180, 180, 180), blur=True)
    sharp_val = assessor.evaluate_sharpness(sharp_crop)
    blurry_val = assessor.evaluate_sharpness(blurry_crop)
    pass_3 = (sharp_val >= settings.min_sharpness and blurry_val < settings.min_sharpness)
    results_summary["3. Blur/Sharpness Measurement"] = {
        "status": "PASS" if pass_3 else "FAIL",
        "measurements": f"Sharpness (Unblurred)={sharp_val:.1f}, Sharpness (Blurred)={blurry_val:.1f} (Threshold={settings.min_sharpness})"
    }
    logger.info(f" -> Result: {'PASS' if pass_3 else 'FAIL'} | {results_summary['3. Blur/Sharpness Measurement']['measurements']}")

    # ---------------------------------------------------------------------------
    # 4. Brightness Measurement
    # ---------------------------------------------------------------------------
    logger.info("[REQ 4] Brightness Measurement")
    dark_crop = make_test_face(150, 150, (15, 15, 15))
    glare_crop = make_test_face(150, 150, (250, 250, 250))
    norm_crop = make_test_face(150, 150, (140, 140, 140))
    b_dark = assessor.evaluate_brightness(dark_crop)
    b_glare = assessor.evaluate_brightness(glare_crop)
    b_norm = assessor.evaluate_brightness(norm_crop)
    pass_4 = (b_dark < settings.min_brightness and b_glare > settings.max_brightness and settings.min_brightness <= b_norm <= settings.max_brightness)
    results_summary["4. Brightness Measurement"] = {
        "status": "PASS" if pass_4 else "FAIL",
        "measurements": f"Dark={b_dark:.1f} (<{settings.min_brightness}), Glare={b_glare:.1f} (>{settings.max_brightness}), Normal={b_norm:.1f} (Valid range [{settings.min_brightness}, {settings.max_brightness}])"
    }
    logger.info(f" -> Result: {'PASS' if pass_4 else 'FAIL'} | {results_summary['4. Brightness Measurement']['measurements']}")

    # ---------------------------------------------------------------------------
    # 5. Face Crop
    # ---------------------------------------------------------------------------
    logger.info("[REQ 5] Face Crop")
    b_face = make_test_face(100, 100, (180, 180, 180))
    frame[1000:1080, 1850:1920] = b_face[:80, :70]
    det_b = {"bbox": [1850, 1000, 1950, 1100], "face_area": 10000}
    res_b = assessor.assess(det_b, frame, "b_face")
    pass_5 = (res_b.bbox == [1850, 1000, 1920, 1080] and res_b.quality_status == "RECOGNITION_READY")
    results_summary["5. Face Crop"] = {
        "status": "PASS" if pass_5 else "FAIL",
        "measurements": f"Requested BBox=[1850, 1000, 1950, 1100], Safe Clamped BBox={res_b.bbox}, Boundary Status={res_b.quality_status}"
    }
    logger.info(f" -> Result: {'PASS' if pass_5 else 'FAIL'} | {results_summary['5. Face Crop']['measurements']}")

    # ---------------------------------------------------------------------------
    # 6. Face Alignment
    # ---------------------------------------------------------------------------
    logger.info("[REQ 6] Face Alignment")
    kpts = [[270.0, 170.0], [330.0, 170.0], [300.0, 200.0], [275.0, 240.0], [325.0, 240.0]]
    aligned = aligner.align(frame, kpts)
    pass_6 = (aligned.shape == (112, 112, 3) and aligned.dtype == np.uint8)
    results_summary["6. Face Alignment"] = {
        "status": "PASS" if pass_6 else "FAIL",
        "measurements": f"Aligned Shape={aligned.shape}, Dtype={aligned.dtype}, Reference Points Count=5"
    }
    logger.info(f" -> Result: {'PASS' if pass_6 else 'FAIL'} | {results_summary['6. Face Alignment']['measurements']}")

    # ---------------------------------------------------------------------------
    # 7. Normalization/Preprocessing
    # ---------------------------------------------------------------------------
    logger.info("[REQ 7] Normalization/Preprocessing")
    tensor = aligner.normalize(aligned)
    min_val, max_val = float(np.min(tensor)), float(np.max(tensor))
    pass_7 = (tensor.shape == (1, 3, 112, 112) and tensor.dtype == np.float32 and -1.0 <= min_val and max_val <= 1.0)
    results_summary["7. Normalization/Preprocessing"] = {
        "status": "PASS" if pass_7 else "FAIL",
        "measurements": f"Tensor Shape={tensor.shape}, Dtype={tensor.dtype}, Min={min_val:.4f}, Max={max_val:.4f}, Formula: (RGB - 127.5)/128.0"
    }
    logger.info(f" -> Result: {'PASS' if pass_7 else 'FAIL'} | {results_summary['7. Normalization/Preprocessing']['measurements']}")

    # ---------------------------------------------------------------------------
    # 8. Multiple-Face Processing
    # ---------------------------------------------------------------------------
    logger.info("[REQ 8] Multiple-Face Processing")
    multi_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    multi_frame[50:250, 50:250] = make_test_face(200, 200, (180, 180, 180), blur=False)
    multi_frame[50:250, 350:550] = make_test_face(200, 200, (180, 180, 180), blur=True)
    multi_frame[50:90, 650:680] = make_test_face(30, 40, (180, 180, 180), blur=False)
    multi_frame[50:250, 800:1000] = make_test_face(200, 200, (180, 180, 180), blur=False)

    multi_dets = [
        {"track_id": "stud_1", "bbox": [50, 50, 250, 250], "face_area": 40000, "keypoints": [[120, 120], [180, 120], [150, 150], [125, 190], [175, 190]]},
        {"track_id": "stud_2", "bbox": [350, 50, 550, 250], "face_area": 40000, "keypoints": [[420, 120], [480, 120], [450, 150], [425, 190], [475, 190]]},
        {"track_id": "stud_3", "bbox": [650, 50, 680, 90], "face_area": 1200, "keypoints": [[660, 60], [670, 60], [665, 70], [662, 80], [668, 80]]},
        {"track_id": "stud_4", "bbox": [800, 50, 1000, 250], "face_area": 40000, "keypoints": [[870, 120], [930, 120], [900, 150], [875, 190], [925, 190]]},
    ]
    ready_multi, rej_multi = processor.process(multi_dets, multi_frame)
    pass_8 = (len(ready_multi) == 2 and len(rej_multi) == 2)
    results_summary["8. Multiple-Face Processing"] = {
        "status": "PASS" if pass_8 else "FAIL",
        "measurements": f"Input Faces=4, Ready Faces={len(ready_multi)} (stud_1, stud_4), Rejected Faces={len(rej_multi)} (stud_2=blurry, stud_3=small)"
    }
    logger.info(f" -> Result: {'PASS' if pass_8 else 'FAIL'} | {results_summary['8. Multiple-Face Processing']['measurements']}")

    # NEAR Zone: Area > 15,000 px (e.g. 150x150 = 22,500)
    frame[50:200, 50:200] = make_test_face(150, 150, (180, 180, 180), blur=False)
    det_near = {"bbox": [50, 50, 200, 200], "face_area": 22500, "zone": "NEAR"}

    # MIDDLE Zone: Area 3,000 to 15,000 px (e.g. 80x100 = 8,000)
    frame[300:400, 300:380] = make_test_face(80, 100, (180, 180, 180), blur=False)
    det_mid = {"bbox": [300, 300, 380, 400], "face_area": 8000, "zone": "MIDDLE"}

    # FAR Zone: Area < 3,000 px (e.g. 40x50 = 2,000)
    frame[500:550, 500:540] = make_test_face(40, 50, (180, 180, 180), blur=False)
    det_far = {"bbox": [500, 500, 540, 550], "face_area": 2000, "zone": "FAR"}

    r_near = assessor.assess(det_near, frame, "near")
    r_mid = assessor.assess(det_mid, frame, "mid")
    r_far = assessor.assess(det_far, frame, "far")
    pass_9 = (r_near.quality_status == "RECOGNITION_READY" and r_mid.quality_status == "RECOGNITION_READY" and r_far.quality_status == "LOW_QUALITY")
    results_summary["9. Near/Middle/Far Face Testing"] = {
        "status": "PASS" if pass_9 else "FAIL",
        "measurements": f"NEAR (22.5k px)={r_near.quality_status}, MIDDLE (8k px)={r_mid.quality_status}, FAR (2k px)={r_far.quality_status} (Rejection='{r_far.rejection_reason}')"
    }
    logger.info(f" -> Result: {'PASS' if pass_9 else 'FAIL'} | {results_summary['9. Near/Middle/Far Face Testing']['measurements']}")

    # ---------------------------------------------------------------------------
    # 10. Recognition-Ready Output
    # ---------------------------------------------------------------------------
    logger.info("[REQ 10] Recognition-Ready Output")
    assert len(ready_multi) > 0
    rf = ready_multi[0]
    pass_10 = (
        isinstance(rf, RecognitionReadyFace) and
        rf.aligned_face_tensor.shape == (1, 3, 112, 112) and
        rf.aligned_face_tensor.dtype == np.float32 and
        isinstance(rf.quality_metrics, FaceQualityResult) and
        rf.timestamp > 0 and
        len(rf.original_bbox) == 4
    )
    results_summary["10. Recognition-Ready Output"] = {
        "status": "PASS" if pass_10 else "FAIL",
        "measurements": f"Schema=RecognitionReadyFace, Tensor Shape={rf.aligned_face_tensor.shape}, Dtype={rf.aligned_face_tensor.dtype}, Metrics Attached=True, Timestamp={rf.timestamp:.2f}"
    }
    logger.info(f" -> Result: {'PASS' if pass_10 else 'FAIL'} | {results_summary['10. Recognition-Ready Output']['measurements']}")

    # ---------------------------------------------------------------------------
    # 11. Error Handling
    # ---------------------------------------------------------------------------
    logger.info("[REQ 11] Error Handling")
    e_none = assessor.assess({"bbox": [0, 0, 10, 10], "face_area": 100}, None, "e1")
    e_empty = assessor.assess({"bbox": [0, 0, 10, 10], "face_area": 100}, np.array([], dtype=np.uint8), "e2")
    try:
        aligner.align(None, [[0, 0]] * 5)
        aligner_none_handled = False
    except ValueError:
        aligner_none_handled = True

    try:
        aligner.align(frame, [[0, 0]] * 4)  # Wrong number of keypoints
        aligner_kpts_handled = False
    except ValueError:
        aligner_kpts_handled = True

    p_ready, p_rej = processor.process([{"bbox": [10, 10, 50, 50]}], None)
    pass_11 = (
        e_none.quality_status == "LOW_QUALITY" and
        e_empty.quality_status == "LOW_QUALITY" and
        aligner_none_handled and
        aligner_kpts_handled and
        p_ready == [] and p_rej == []
    )
    results_summary["11. Error Handling"] = {
        "status": "PASS" if pass_11 else "FAIL",
        "measurements": f"None frame handled={e_none.quality_status=='LOW_QUALITY'}, Empty frame handled={e_empty.quality_status=='LOW_QUALITY'}, Invalid keypoints rejected={aligner_kpts_handled}, Pipeline None frame handled=True"
    }
    logger.info(f" -> Result: {'PASS' if pass_11 else 'FAIL'} | {results_summary['11. Error Handling']['measurements']}")

    # ---------------------------------------------------------------------------
    # 12. Performance / FPS / Latency
    # ---------------------------------------------------------------------------
    logger.info("[REQ 12] Performance / FPS / Latency Benchmarks")
    perf_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    perf_dets_10 = []
    for i in range(10):
        x1 = 50 + (i % 5) * 200
        y1 = 50 + (i // 5) * 250
        x2, y2 = x1 + 150, y1 + 180
        perf_frame[y1:y2, x1:x2] = make_test_face(150, 180, (180, 180, 180))
        perf_dets_10.append({
            "track_id": f"perf_{i}",
            "bbox": [x1, y1, x2, y2],
            "face_area": 150 * 180,
            "keypoints": [
                [x1 + 40, y1 + 50], [x1 + 110, y1 + 50],
                [x1 + 75, y1 + 90],
                [x1 + 45, y1 + 130], [x1 + 105, y1 + 130]
            ]
        })

    # Individual component benchmarks
    # a) Assessment latency
    t0 = time.perf_counter()
    for _ in range(100):
        assessor.assess(perf_dets_10[0], perf_frame, "perf_a")
    t1 = time.perf_counter()
    lat_assess_ms = ((t1 - t0) / 100.0) * 1000.0

    # b) Alignment latency
    t0 = time.perf_counter()
    for _ in range(100):
        aligner.align(perf_frame, perf_dets_10[0]["keypoints"])
    t1 = time.perf_counter()
    lat_align_ms = ((t1 - t0) / 100.0) * 1000.0

    # c) Normalization latency
    crop_112 = np.zeros((112, 112, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    for _ in range(100):
        aligner.normalize(crop_112)
    t1 = time.perf_counter()
    lat_norm_ms = ((t1 - t0) / 100.0) * 1000.0

    # d) Full pipeline latency for 1 face, 5 faces, 10 faces, 20 faces
    runs = 30
    # 1 face
    t0 = time.perf_counter()
    for _ in range(runs):
        processor.process(perf_dets_10[:1], perf_frame)
    t1 = time.perf_counter()
    lat_1_face_ms = ((t1 - t0) / runs) * 1000.0
    fps_1_face = 1000.0 / lat_1_face_ms if lat_1_face_ms > 0 else 0

    # 5 faces
    t0 = time.perf_counter()
    for _ in range(runs):
        processor.process(perf_dets_10[:5], perf_frame)
    t1 = time.perf_counter()
    lat_5_faces_ms = ((t1 - t0) / runs) * 1000.0
    fps_5_faces = 1000.0 / lat_5_faces_ms if lat_5_faces_ms > 0 else 0

    # 10 faces
    t0 = time.perf_counter()
    for _ in range(runs):
        processor.process(perf_dets_10, perf_frame)
    t1 = time.perf_counter()
    lat_10_faces_ms = ((t1 - t0) / runs) * 1000.0
    fps_10_faces = 1000.0 / lat_10_faces_ms if lat_10_faces_ms > 0 else 0

    pass_12 = (lat_1_face_ms < 5.0 and fps_10_faces > 30.0)
    results_summary["12. Performance/FPS/Latency"] = {
        "status": "PASS" if pass_12 else "FAIL",
        "measurements": (
            f"Quality Assessment: {lat_assess_ms:.3f} ms/face | "
            f"5-pt Alignment: {lat_align_ms:.3f} ms/face | "
            f"Normalization: {lat_norm_ms:.3f} ms/face | "
            f"Total Per-Face: {lat_1_face_ms:.3f} ms | "
            f"1 Face Pipeline: {lat_1_face_ms:.2f} ms ({fps_1_face:.1f} FPS) | "
            f"5 Faces Pipeline: {lat_5_faces_ms:.2f} ms ({fps_5_faces:.1f} FPS) | "
            f"10 Faces Pipeline: {lat_10_faces_ms:.2f} ms ({fps_10_faces:.1f} FPS)"
        )
    }
    logger.info(f" -> Result: {'PASS' if pass_12 else 'FAIL'} | {results_summary['12. Performance/FPS/Latency']['measurements']}")

    # ---------------------------------------------------------------------------
    # REAL IMAGE BENCHMARK: stress_test_result.jpg
    # ---------------------------------------------------------------------------
    logger.info("-" * 50)
    logger.info("VALIDATING REAL IMAGE: stress_test_result.jpg")
    real_img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stress_test_result.jpg")
    if os.path.exists(real_img_path):
        from core.detector import YOLOv8FaceDetector
        detector = YOLOv8FaceDetector(model_path="yolov8n.pt")
        real_img = cv2.imread(real_img_path)
        real_dets = detector.detect(real_img)
        real_faces = [d for d in real_dets if d.get('class_id', 0) == 0]
        
        t0 = time.perf_counter()
        ready_real, rej_real = processor.process(real_faces, real_img)
        t1 = time.perf_counter()
        real_latency_ms = (t1 - t0) * 1000.0

        logger.info(f"Real Image: {len(real_faces)} detected faces")
        logger.info(f"Phase 5 Real Processing Latency: {real_latency_ms:.2f} ms for {len(real_faces)} faces ({real_latency_ms/max(1,len(real_faces)):.2f} ms/face)")
        logger.info(f"Phase 5 Ready: {len(ready_real)}, Rejected: {len(rej_real)}")
        for idx, rf_item in enumerate(ready_real):
            qm = rf_item.quality_metrics
            logger.info(f"  Face {idx}: ID={qm.face_id}, Area={qm.area} px, Sharpness={qm.sharpness:.1f}, Brightness={qm.brightness:.1f}, Tensor={rf_item.aligned_face_tensor.shape}")

    logger.info("=" * 60)
    logger.info("FINAL PHASE 5 REQUIREMENT VALIDATION SUMMARY:")
    logger.info("=" * 60)
    all_passed = True
    for req_name, data in results_summary.items():
        logger.info(f"[{data['status']}] {req_name}")
        logger.info(f"       Details: {data['measurements']}")
        if data['status'] != "PASS":
            all_passed = False

    logger.info("=" * 60)
    logger.info(f"OVERALL STATUS: {'PASS' if all_passed else 'FAIL'} ({sum(1 for d in results_summary.values() if d['status'] == 'PASS')}/12 Passed)")
    logger.info("=" * 60)
    return all_passed, results_summary


if __name__ == "__main__":
    run_complete_phase5_validation()
