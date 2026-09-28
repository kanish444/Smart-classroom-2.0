import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from core.detector import YOLOv8FaceDetector

def test_tiling():
    scene = cv2.imread("scratch/realistic_classroom_scene.jpg")
    h, w = scene.shape[:2]
    
    det = YOLOv8FaceDetector()
    
    # Define upper classroom zone (where distant students sit: y: 100..450)
    # Split into 2 overlapping tiles:
    # Tile 1: x: 0..760, y: 100..460 (size: 760x360)
    # Tile 2: x: 520..1280, y: 100..460 (size: 760x360)
    # Overlap is 240px (760 - 520 = 240px)
    tiles = [
        (0, 100, 760, 460),
        (520, 100, 1280, 460)
    ]
    
    print("\n--- BENCHMARK: Tiled Detection ---")
    t0 = time.perf_counter()
    all_dets = []
    
    for tx1, ty1, tx2, ty2 in tiles:
        tile_crop = scene[ty1:ty2, tx1:tx2]
        tile_dets = det.detect(tile_crop)
        # Map coordinates back
        for td in tile_dets:
            bx1, by1, bx2, by2 = td["bbox"]
            mapped_bbox = [bx1 + tx1, by1 + ty1, bx2 + tx1, by2 + ty1]
            mapped_crop_bbox = [td["crop_bbox"][0] + tx1, td["crop_bbox"][1] + ty1, td["crop_bbox"][2] + tx1, td["crop_bbox"][3] + ty1]
            mapped_kpts = [[pt[0] + tx1, pt[1] + ty1] for pt in td["keypoints"]]
            
            all_dets.append({
                "bbox": mapped_bbox,
                "crop_bbox": mapped_crop_bbox,
                "confidence": td["confidence"],
                "face_area": (mapped_bbox[2] - mapped_bbox[0]) * (mapped_bbox[3] - mapped_bbox[1]),
                "zone": td["zone"],
                "class_id": td["class_id"],
                "keypoints": mapped_kpts
            })
            
    t1 = time.perf_counter()
    print(f"2 Tiles total latency: {(t1 - t0)*1000:.1f}ms | Raw detections from tiles: {len(all_dets)}")
    for d in all_dets:
        bx = d['bbox']
        bw = bx[2] - bx[0]
        bh = bx[3] - bx[1]
        print(f"  Tile det: {bx} ({bw}x{bh}), Conf: {d['confidence']:.2f}")

if __name__ == "__main__":
    test_tiling()
