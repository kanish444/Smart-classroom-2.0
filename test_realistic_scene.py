import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from core.detector import YOLOv8FaceDetector

def test_actual_face_crops():
    # 1. Load full enrollment frame and crop the actual face
    sample_path = "data/enrollment/922524243069/sample_01.jpg"
    full_frame = cv2.imread(sample_path)
    # Actual face coordinates from detection: [419, 193, 654, 480]
    face_crop = full_frame[193:480, 419:654] # ~287x235 px
    fh, fw = face_crop.shape[:2]
    print(f"Extracted genuine face crop: {fw}x{fh}")

    # Build a realistic 1280x720 classroom image with background students
    # Classroom background
    scene = np.full((720, 1280, 3), (190, 195, 200), dtype=np.uint8)
    # Classroom floor and walls
    cv2.rectangle(scene, (0, 350), (1280, 720), (150, 155, 160), -1)
    # Desks / whiteboard
    cv2.rectangle(scene, (200, 50), (1080, 280), (40, 55, 45), -1) # board
    # Windows on right side creating backlight/glare
    cv2.rectangle(scene, (1100, 30), (1280, 350), (240, 245, 255), -1)

    # Place Face 1: CASE A - Large Close Face (Near camera, 180x180 px at x=200, y=380)
    f_large = cv2.resize(face_crop, (180, 180))
    scene[380:560, 200:380] = f_large

    # Place Face 2: Medium Face (Middle row student, 80x80 px at x=600, y=300)
    f_med = cv2.resize(face_crop, (80, 80))
    scene[300:380, 600:680] = f_med

    # Place Face 3: CASE C - Small Distant Face (Back row student, 40x40 px at x=400, y=200)
    f_small = cv2.resize(face_crop, (40, 40))
    scene[200:240, 400:440] = f_small

    # Place Face 4: Tiny Distant Face (Far back corner, 28x28 px at x=800, y=180)
    f_tiny = cv2.resize(face_crop, (28, 28))
    scene[180:208, 800:828] = f_tiny

    # Place Face 5: CASE B - Partially Visible Edge Face (Right edge: 70x70 px face, only left 35px in frame)
    f_part = cv2.resize(face_crop, (70, 70))
    scene[280:350, 1245:1280] = f_part[:, :35]

    # Place Face 6: Back of head (hair texture, no face features, 80x80 at x=880, y=340)
    f_back = np.full((80, 80, 3), (30, 25, 20), dtype=np.uint8)
    cv2.circle(f_back, (40, 40), 38, (15, 12, 10), -1)
    scene[340:420, 880:960] = f_back

    # Place Face 7: Backlight / Glare student (70x70 washed out face near window at x=1050, y=220)
    f_glare = cv2.addWeighted(cv2.resize(face_crop, (70, 70)), 0.45, np.full((70, 70, 3), 255, dtype=np.uint8), 0.55, 0)
    scene[220:290, 1050:1120] = f_glare

    cv2.imwrite("scratch/realistic_classroom_scene.jpg", scene)
    print("Realistic classroom scene saved to scratch/realistic_classroom_scene.jpg")

    det = YOLOv8FaceDetector()

    for conf in [0.35, 0.25, 0.20]:
        for imgsz in [640, 960, 1280]:
            det.confidence_threshold = conf
            det.settings.input_size = imgsz
            t0 = time.perf_counter()
            dets = det.detect(scene)
            t1 = time.perf_counter()
            lat = (t1 - t0) * 1000
            print(f"\nConf={conf:.2f} | imgsz={imgsz} | Latency={lat:.1f}ms | Detected: {len(dets)}")
            for d in dets:
                bx = d['bbox']
                bw = bx[2] - bx[0]
                bh = bx[3] - bx[1]
                print(f"  Box: {bx} ({bw}x{bh}), Conf: {d['confidence']:.2f}, Zone: {d['zone']}")

if __name__ == "__main__":
    test_actual_face_crops()
