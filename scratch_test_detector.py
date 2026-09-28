import os
import sys
import time
import cv2
import numpy as np

# Ensure root dir in path
sys.path.insert(0, os.path.abspath("."))

from core.detector import YOLOv8FaceDetector

def create_classroom_test_scene():
    """
    Creates a synthetic 1280x720 classroom test frame using real enrolled student face crops:
    - Large close-up face (center-left) ~180x180 px
    - Medium face (middle row) ~80x80 px
    - Small distant face (back row) ~32x32 px
    - Very small distant face (far corner) ~22x22 px
    - Partially visible face (right edge, 50% clipped) ~70x70 px
    - Back of head / hair texture ~80x80 px
    - Glare / washed out face ~90x90 px
    """
    # Background classroom-like image (walls, lighting gradient)
    frame = np.full((720, 1280, 3), 180, dtype=np.uint8)
    # Add ceiling lights / window glare on top right
    cv2.rectangle(frame, (800, 0), (1280, 250), (245, 245, 250), -1)
    # Add blackboard / desk area
    cv2.rectangle(frame, (100, 100), (700, 350), (45, 60, 45), -1)

    # Load real face image from enrollment
    sample_path = "data/enrollment/922524243069/sample_01.jpg"
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Missing sample face at {sample_path}")
    face_img = cv2.imread(sample_path)

    # 1. Large Face (Near): 180x180 at (150, 400)
    face_large = cv2.resize(face_img, (180, 180))
    frame[400:580, 150:330] = face_large

    # 2. Medium Face (Middle): 80x80 at (500, 320)
    face_med = cv2.resize(face_img, (80, 80))
    frame[320:400, 500:580] = face_med

    # 3. Small Face (Far Row): 36x36 at (720, 220)
    face_small = cv2.resize(face_img, (36, 36))
    frame[220:256, 720:756] = face_small

    # 4. Tiny Distant Face (Back Row): 24x24 at (350, 200)
    face_tiny = cv2.resize(face_img, (24, 24))
    frame[200:224, 350:374] = face_tiny

    # 5. Partially Visible Edge Face: 80x80 placed so half is off-screen on right edge
    face_edge = cv2.resize(face_img, (80, 80))
    # Place at x=1240..1280 (only left 40px visible)
    frame[300:380, 1240:1280] = face_edge[:, :40]

    # 6. Glare / Overexposed Face: 80x80 with high brightness / washed out contrast at (950, 260)
    face_glare = cv2.addWeighted(face_med, 0.4, np.full((80, 80, 3), 255, dtype=np.uint8), 0.6, 0)
    frame[260:340, 950:1030] = face_glare

    # 7. Back of Head (Hair texture, no facial features) at (250, 280)
    hair = np.full((70, 70, 3), (25, 20, 15), dtype=np.uint8)
    cv2.circle(hair, (35, 35), 32, (15, 10, 5), -1)
    frame[280:350, 250:320] = hair

    return frame

def run_experiment():
    frame = create_classroom_test_scene()
    os.makedirs("scratch", exist_ok=True)
    cv2.imwrite("scratch/classroom_test_scene.jpg", frame)
    print("Classroom test scene saved to scratch/classroom_test_scene.jpg (1280x720)")

    detector = YOLOv8FaceDetector()

    # Test 1: Current default (imgsz=640, conf=0.35)
    print("\n--- TEST 1: Default imgsz=640, conf=0.35 ---")
    t0 = time.perf_counter()
    dets_640 = detector.detect(frame)
    t1 = time.perf_counter()
    lat_640 = (t1 - t0) * 1000
    print(f"Latency: {lat_640:.1f}ms | Faces detected: {len(dets_640)}")
    for d in dets_640:
        w = d['bbox'][2] - d['bbox'][0]
        h = d['bbox'][3] - d['bbox'][1]
        print(f"  Box: {d['bbox']} ({w}x{h}), Conf: {d['confidence']:.2f}, Zone: {d['zone']}")

    # Test 2: imgsz=960, conf=0.25
    print("\n--- TEST 2: imgsz=960, conf=0.25 ---")
    detector.confidence_threshold = 0.25
    detector.settings.input_size = 960
    t0 = time.perf_counter()
    dets_960 = detector.detect(frame)
    t1 = time.perf_counter()
    lat_960 = (t1 - t0) * 1000
    print(f"Latency: {lat_960:.1f}ms | Faces detected: {len(dets_960)}")
    for d in dets_960:
        w = d['bbox'][2] - d['bbox'][0]
        h = d['bbox'][3] - d['bbox'][1]
        print(f"  Box: {d['bbox']} ({w}x{h}), Conf: {d['confidence']:.2f}, Zone: {d['zone']}")

    # Test 3: imgsz=1280, conf=0.25
    print("\n--- TEST 3: imgsz=1280, conf=0.25 ---")
    detector.settings.input_size = 1280
    t0 = time.perf_counter()
    dets_1280 = detector.detect(frame)
    t1 = time.perf_counter()
    lat_1280 = (t1 - t0) * 1000
    print(f"Latency: {lat_1280:.1f}ms | Faces detected: {len(dets_1280)}")
    for d in dets_1280:
        w = d['bbox'][2] - d['bbox'][0]
        h = d['bbox'][3] - d['bbox'][1]
        print(f"  Box: {d['bbox']} ({w}x{h}), Conf: {d['confidence']:.2f}, Zone: {d['zone']}")

if __name__ == "__main__":
    run_experiment()
