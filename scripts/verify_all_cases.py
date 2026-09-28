import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from core.tracking_pipeline import TrackingPipeline
from core.schemas import RecognitionStatus
from app.state import get_app_state

def run_system_verification():
    print("=" * 70)
    print("SMARTCLASS VISION AI - COMPREHENSIVE PIPELINE VERIFICATION")
    print("=" * 70)

    # 1. Load sample enrolled face
    sample_path = "data/enrollment/922524243069/sample_01.jpg"
    full_frame = cv2.imread(sample_path)
    # Real face crop: [419, 193, 654, 480]
    face_crop = full_frame[193:480, 419:654]
    print(f"[INIT] Loaded enrolled student 922524243069 face crop: {face_crop.shape[1]}x{face_crop.shape[0]} px")

    # Initialize Tracking Pipeline with shared state
    app_state = get_app_state()
    v_store = app_state.get_vector_store()
    print(f"[INIT] FAISS Vector Store total vectors: {v_store.total_vectors}")

    pipeline = TrackingPipeline()
    pipeline.reset()

    # -------------------------------------------------------------
    # BUILD TEST CASES IN A REALISTIC 1280x720 CLASSROOM FRAME
    # -------------------------------------------------------------
    # Background
    frame = np.full((720, 1280, 3), (190, 195, 200), dtype=np.uint8)
    # Floor
    cv2.rectangle(frame, (0, 380), (1280, 720), (145, 150, 155), -1)
    # Blackboard / desks
    cv2.rectangle(frame, (180, 40), (1050, 260), (40, 55, 45), -1)
    # Windows on right side creating backlight/glare
    cv2.rectangle(frame, (1100, 20), (1280, 320), (245, 250, 255), -1)

    # TEST 1 & CASE A: Large Close Face (x=160, y=360, size 180x180 px) - Enrolled Student 922524243069
    f_large = cv2.resize(face_crop, (180, 180))
    frame[360:540, 160:340] = f_large

    # TEST 2: Medium Distance Face (x=550, y=280, size 80x80 px) - Enrolled Student 922524243069
    f_med = cv2.resize(face_crop, (80, 80))
    frame[280:360, 550:630] = f_med

    # TEST 3 & CASE C: Small / Distant Face in Background (x=380, y=190, size 42x42 px)
    f_small = cv2.resize(face_crop, (42, 42))
    frame[190:232, 380:422] = f_small

    # Tiny Face in Far Corner (x=780, y=170, size 26x26 px - genuinely too small for ArcFace)
    f_tiny = cv2.resize(face_crop, (26, 26))
    frame[170:196, 780:806] = f_tiny

    # TEST 4 & CASE B: Partially Visible Face (x=1250..1280, only left 30px visible)
    f_part = cv2.resize(face_crop, (70, 70))
    frame[270:340, 1250:1280] = f_part[:, :30]

    # TEST 6: Glare / Backlight Washout Face near window (x=1030, y=210, 70x70 px)
    f_glare = cv2.addWeighted(cv2.resize(face_crop, (70, 70)), 0.35, np.full((70, 70, 3), 255, dtype=np.uint8), 0.65, 0)
    frame[210:280, 1030:1100] = f_glare

    # TEST 8: Back of Head (No facial features, 70x70 px at x=880, y=320)
    f_back = np.full((70, 70, 3), (25, 20, 15), dtype=np.uint8)
    cv2.circle(f_back, (35, 35), 32, (15, 10, 5), -1)
    frame[320:390, 880:950] = f_back

    # -------------------------------------------------------------
    # TEMPORAL STABILIZATION OVER MULTIPLE FRAMES
    # -------------------------------------------------------------
    print("\n[RUNNING 5 CONSECUTIVE FRAMES THROUGH BYTE-TRACK & TEMPORAL STABILIZER]")
    all_seen_tracks = []
    for f_idx in range(1, 6):
        t0 = time.perf_counter()
        tracks = pipeline.process_frame(frame)
        all_seen_tracks.extend(tracks)
        t_el = (time.perf_counter() - t0) * 1000.0

        print(f"\n--- FRAME {f_idx} (Pipeline Latency: {t_el:.1f}ms, Tracks: {len(tracks)}) ---")
        for t in tracks:
            bx = t.bbox
            bw = bx[2] - bx[0]
            bh = bx[3] - bx[1]
            diag = t.diagnostics
            print(
                f"  Track {t.track_id:2d} | BBox: [{bx[0]:4d},{bx[1]:4d},{bx[2]:4d},{bx[3]:4d}] ({bw:3d}x{bh:3d}) | "
                f"DetConf: {t.score:.2f} | Display: {t.display_status:14s} | "
                f"ID: {str(t.stable_student_id):16s} | Sim: {t.current_similarity:.2f} | "
                f"Qual: {t.quality_status:16s} | Reason: {diag.get('rejection_reason') or 'OK'}"
            )

    # -------------------------------------------------------------
    # VALIDATE EACH TEST CASE AGAINST ACCEPTANCE CRITERIA
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERIFICATION OF ACCEPTANCE CRITERIA:")
    print("=" * 70)

    # Find the tracks by position
    def find_track_near(x, y, w_expected):
        for tr in reversed(all_seen_tracks):
            cx = (tr.bbox[0] + tr.bbox[2]) / 2
            cy = (tr.bbox[1] + tr.bbox[3]) / 2
            if abs(cx - x) < 60 and abs(cy - y) < 60:
                return tr
        return None

    tr_large = find_track_near(250, 450, 180)
    tr_med = find_track_near(590, 320, 80)
    tr_small = find_track_near(400, 210, 42)
    tr_tiny = find_track_near(790, 180, 26)
    tr_part = find_track_near(1265, 305, 30)
    tr_glare = find_track_near(1065, 245, 70)
    tr_back = find_track_near(915, 355, 70)

    results = {}

    # Case A: Large close face -> RECOGNIZED
    if tr_large and tr_large.stable_student_id == "STU_922524243069" and tr_large.display_status == "RECOGNIZED":
        results["CASE A (Large Close Face)"] = "PASSED: Detected and correctly recognized as STU_922524243069"
    else:
        results["CASE A (Large Close Face)"] = f"FAILED: {tr_large}"

    # Test 2: Medium distance face -> RECOGNIZED if quality passes
    if tr_med and tr_med.display_status == "RECOGNIZED":
        results["TEST 2 (Medium Face)"] = f"PASSED: Recognized as {tr_med.stable_student_id} (Sim: {tr_med.current_similarity:.2f})"
    else:
        results["TEST 2 (Medium Face)"] = f"STATUS: {tr_med.display_status if tr_med else 'Not detected'}"

    # Case C: Small/Distant Face (42x42 px) -> Detected & Evaluated
    if tr_small:
        results["CASE C (Small/Distant Face)"] = f"PASSED: Successfully detected ({tr_small.bbox[2]-tr_small.bbox[0]}x{tr_small.bbox[3]-tr_small.bbox[1]} px), Display: {tr_small.display_status}, Qual: {tr_small.quality_status}"
    else:
        results["CASE C (Small/Distant Face)"] = "FAILED: Not detected"

    # Tiny Face (26x26 px) -> Detected as FACE TOO SMALL, NOT force-matched
    if tr_tiny:
        if tr_tiny.display_status in ("FACE TOO SMALL", "UNKNOWN") and tr_tiny.stable_student_id is None:
            results["TINY FACE (<30px)"] = f"PASSED: Labeled {tr_tiny.display_status}, No false positive forced identity"
        else:
            results["TINY FACE (<30px)"] = f"FAILED: Hallucinated identity {tr_tiny.stable_student_id}"
    else:
        results["TINY FACE (<30px)"] = "PASSED: Below detector threshold, not hallucinated"

    # Case B: Partially Visible Edge Face -> NOT force-matched
    if tr_part:
        if tr_part.stable_student_id is None and tr_part.display_status in ("LOW QUALITY", "UNKNOWN", "FACE TOO SMALL"):
            results["CASE B (Partial Edge Face)"] = f"PASSED: Detected, Gated as {tr_part.display_status} (No forced identity)"
        else:
            results["CASE B (Partial Edge Face)"] = f"FAILED: Force matched {tr_part.stable_student_id}"
    else:
        results["CASE B (Partial Edge Face)"] = "PASSED: Outside safe crop, not falsely recognized"

    # Test 6: Glare / Backlight Face -> Detected with Glare Diagnostics
    if tr_glare:
        results["TEST 6 (Glare / Backlight Face)"] = f"PASSED: Detected, GlareScore: {tr_glare.diagnostics.get('glare_score')}, Brightness: {tr_glare.diagnostics.get('brightness')}, Status: {tr_glare.display_status}"
    else:
        results["TEST 6 (Glare / Backlight Face)"] = "FAILED: Not detected"

    # Test 8: Back of Head -> MUST NOT be recognized as a face
    if tr_back is None:
        results["TEST 8 (Back of Head)"] = "PASSED: Correctly ignored (0 false positive face detections)"
    else:
        results["TEST 8 (Back of Head)"] = f"FAILED: Back of head falsely detected as face: {tr_back}"

    for k, v in results.items():
        print(f"  {k:30s}: {v}")

    # Generate and save rendered HUD frame
    rendered_debug = pipeline.draw_debug_overlay(frame, tracks, fps=28.5)
    cv2.imwrite("scratch/verification_debug_overlay.jpg", rendered_debug)
    print("\n[SAVED] Verification debug overlay saved to scratch/verification_debug_overlay.jpg")

    rendered_normal = app_state._render_normal_view(frame, tracks)
    cv2.imwrite("scratch/verification_normal_overlay.jpg", rendered_normal)
    print("[SAVED] Verification normal overlay saved to scratch/verification_normal_overlay.jpg")
    print("=" * 70)

if __name__ == "__main__":
    run_system_verification()
