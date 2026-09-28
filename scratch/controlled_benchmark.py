import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import cv2
import numpy as np

# 1. Initialize Reference Pipeline
yunet_path = r"C:\Users\kanis\OneDrive\Documents\muliti face detecttion\muliti face detecttion\facevision-ai\backend\models\face_detection_yunet_2023mar.onnx"
sface_path = r"C:\Users\kanis\OneDrive\Documents\muliti face detecttion\muliti face detecttion\facevision-ai\backend\models\face_recognition_sface_2021dec.onnx"

has_ref = os.path.exists(yunet_path) and os.path.exists(sface_path)
print("Reference models found:", has_ref)

# 2. Initialize Our Pipeline
from core.scrfd_detector import SCRFDDetector
from core.face_alignment import FaceAligner
from core.face_quality import FaceQualityAssessor
from core.face_embedder import ArcFaceEmbedder
from core.vector_store import FaissVectorStore
from core.recognizer import FaceRecognizer
from core.byte_tracker import ByteTracker
from core.schemas import RecognitionReadyFace

our_detector = SCRFDDetector()
our_aligner = FaceAligner()
our_assessor = FaceQualityAssessor()
our_embedder = ArcFaceEmbedder()
our_vector_store = FaissVectorStore(index_path="database/faiss_index.bin")
our_recognizer = FaceRecognizer(embedder=our_embedder, vector_store=our_vector_store)
our_tracker = ByteTracker()

# Enrolled student for test
known_img_path = "data/enrollment/922524243069/sample_01.jpg"
known_img = cv2.imread(known_img_path)
classroom_img = cv2.imread("scratch/realistic_classroom_scene.jpg")

# Pre-enroll known student in reference pipeline
ref_detector = cv2.FaceDetectorYN.create(
    model=yunet_path,
    config="",
    input_size=(known_img.shape[1], known_img.shape[0]),
    score_threshold=0.45,
    nms_threshold=0.30
)
ref_recognizer = cv2.FaceRecognizerSF.create(
    model=sface_path,
    config=""
)

_, ref_faces = ref_detector.detect(known_img)
ref_known_emb = None
if ref_faces is not None and len(ref_faces) > 0:
    ref_aligned = ref_recognizer.alignCrop(known_img, ref_faces[0])
    ref_feat = ref_recognizer.feature(ref_aligned).flatten().astype(np.float32)
    ref_norm = np.linalg.norm(ref_feat)
    if ref_norm > 1e-6:
        ref_known_emb = ref_feat / ref_norm
print("Reference enrolled known embedding extracted:", ref_known_emb is not None)

def run_ref_pipeline(img, target_name="kanish"):
    h, w = img.shape[:2]
    ref_detector.setInputSize((w, h))
    t0 = time.perf_counter()
    _, faces = ref_detector.detect(img)
    t_det = (time.perf_counter() - t0) * 1000.0

    dets_out = []
    if faces is not None:
        for f in faces:
            bx = int(max(0, f[0])), int(max(0, f[1])), int(f[2]), int(f[3])
            score = float(f[14])
            t_rec_0 = time.perf_counter()
            aligned = ref_recognizer.alignCrop(img, f)
            feat = ref_recognizer.feature(aligned).flatten().astype(np.float32)
            norm = np.linalg.norm(feat)
            if norm > 1e-6:
                feat /= norm
            sim = float(np.dot(ref_known_emb, feat)) if ref_known_emb is not None else 0.0
            t_rec = (time.perf_counter() - t_rec_0) * 1000.0
            status = "MATCH" if sim >= 0.363 else "UNKNOWN"
            dets_out.append({
                "bbox": [bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3]],
                "dim": (bx[2], bx[3]),
                "conf": score,
                "sim": sim,
                "status": status,
                "rec_time": t_rec
            })
    return {
        "num_faces": len(dets_out),
        "det_ms": t_det,
        "faces": dets_out
    }

def run_our_pipeline(img):
    t0 = time.perf_counter()
    raw = our_detector.detect(img)
    t_det = (time.perf_counter() - t0) * 1000.0

    dets_out = []
    for d in raw:
        t_rec_0 = time.perf_counter()
        q = our_assessor.assess(d, img, "f")
        t_tensor = our_aligner.align_and_normalize(img, d["keypoints"])
        ready = RecognitionReadyFace(
            quality_metrics=q,
            original_bbox=d["bbox"],
            aligned_face_tensor=t_tensor,
            timestamp=time.time()
        )
        rec = our_recognizer.recognize_face(ready)
        t_rec = (time.perf_counter() - t_rec_0) * 1000.0
        bx = d["bbox"]
        dets_out.append({
            "bbox": bx,
            "dim": (bx[2] - bx[0], bx[3] - bx[1]),
            "conf": d["confidence"],
            "sim": rec.similarity,
            "margin": rec.margin,
            "status": rec.status,
            "matched_id": rec.matched_student_id,
            "quality_state": q.quality_state,
            "quality_score": q.quality_score,
            "rec_time": t_rec
        })
    return {
        "num_faces": len(dets_out),
        "det_ms": t_det,
        "faces": dets_out
    }

print("\n" + "="*70)
print("BENCHMARK TEST 1: SINGLE LARGE KNOWN FACE (1280x720 frame)")
print("="*70)
ref_res1 = run_ref_pipeline(known_img)
our_res1 = run_our_pipeline(known_img)
print(f"REFERENCE: Faces={ref_res1['num_faces']} | Det Latency={ref_res1['det_ms']:.1f}ms")
for f in ref_res1["faces"]:
    print(f"   Box={f['bbox']} ({f['dim'][0]}x{f['dim'][1]}px) | Conf={f['conf']:.3f} | Sim={f['sim']:.3f} | Status={f['status']}")
print(f"OUR PIPELINE: Faces={our_res1['num_faces']} | Det Latency={our_res1['det_ms']:.1f}ms")
for f in our_res1["faces"]:
    print(f"   Box={f['bbox']} ({f['dim'][0]}x{f['dim'][1]}px) | Conf={f['conf']:.3f} | Sim={f['sim']:.3f} | Margin={f['margin']:.3f} | Status={f['status']} ({f['matched_id']}) | Quality={f['quality_state']}")

print("\n" + "="*70)
print("BENCHMARK TEST 2: CROWDED / MULTI-SCALE CLASSROOM SCENE (1280x720)")
print("="*70)
ref_res2 = run_ref_pipeline(classroom_img)
our_res2 = run_our_pipeline(classroom_img)
print(f"REFERENCE: Faces={ref_res2['num_faces']} | Det Latency={ref_res2['det_ms']:.1f}ms")
for f in ref_res2["faces"]:
    print(f"   Box={f['bbox']} ({f['dim'][0]}x{f['dim'][1]}px) | Conf={f['conf']:.3f} | Sim={f['sim']:.3f} | Status={f['status']}")
print(f"OUR PIPELINE: Faces={our_res2['num_faces']} | Det Latency={our_res2['det_ms']:.1f}ms")
for f in our_res2["faces"]:
    print(f"   Box={f['bbox']} ({f['dim'][0]}x{f['dim'][1]}px) | Conf={f['conf']:.3f} | Sim={f['sim']:.3f} | Margin={f['margin']:.3f} | Status={f['status']} ({f['matched_id']}) | Quality={f['quality_state']}")

print("\n" + "="*70)
print("BENCHMARK TEST 3: UNKNOWN RANDOM / NON-REGISTERED FACE")
print("="*70)
unknown_scene = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
ref_res3 = run_ref_pipeline(unknown_scene)
our_res3 = run_our_pipeline(unknown_scene)
print(f"REFERENCE: False Detections={ref_res3['num_faces']}")
print(f"OUR PIPELINE: False Detections={our_res3['num_faces']}")
