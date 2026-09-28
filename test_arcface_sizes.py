import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from core.face_alignment import FaceAligner
from core.face_embedder import ArcFaceEmbedder
from core.vector_store import FaissVectorStore

def test_arcface_on_sizes():
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder()
    v_store = FaissVectorStore()
    
    print(f"Total enrolled vectors in FAISS: {v_store.total_vectors}")
    
    # Load original face
    sample_path = "data/enrollment/922524243069/sample_01.jpg"
    full_frame = cv2.imread(sample_path)
    # Face crop [419, 193, 654, 480]
    face_crop = full_frame[193:480, 419:654]
    
    # Synthetic keypoints for alignment
    def get_kpts(w, h):
        return [
            [w * 0.31, h * 0.38],
            [w * 0.69, h * 0.38],
            [w * 0.50, h * 0.58],
            [w * 0.34, h * 0.78],
            [w * 0.66, h * 0.78],
        ]
        
    sizes = [180, 120, 80, 60, 50, 45, 40, 35, 30, 25, 20]
    print("\n--- ArcFace Similarity vs Face Size (Original student 922524243069) ---")
    for s in sizes:
        resized = cv2.resize(face_crop, (s, s))
        kpts = get_kpts(s, s)
        try:
            tensor = aligner.align_and_normalize(resized, kpts)
            emb = embedder.generate_embedding(tensor)
            matches = v_store.search(emb, top_k=1)
            if matches:
                sim, meta = matches[0]
                stu_id = meta.get("student_id")
                print(f"Size: {s:3d}x{s:3d} (Area: {s*s:5d}) | Sim: {sim:.3f} | Matched: {stu_id} | >=0.65? {sim >= 0.65}")
            else:
                print(f"Size: {s:3d}x{s:3d} | No FAISS match")
        except Exception as e:
            print(f"Size: {s:3d}x{s:3d} | Error: {e}")

if __name__ == "__main__":
    test_arcface_on_sizes()
