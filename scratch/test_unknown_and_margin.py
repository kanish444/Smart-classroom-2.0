import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np

from core.face_alignment import FaceAligner
from core.face_quality import FaceQualityAssessor
from core.face_embedder import ArcFaceEmbedder
from core.vector_store import FaissVectorStore
from core.recognizer import FaceRecognizer
from core.schemas import RecognitionReadyFace, FaceQualityResult, RecognitionStatus

embedder = ArcFaceEmbedder()
vector_store = FaissVectorStore(index_path="database/faiss_index.bin")
recognizer = FaceRecognizer(embedder=embedder, vector_store=vector_store, threshold=0.50, margin_threshold=0.08)

def test_impostor_rejection():
    # 1. Load registered student 922524243069 sample
    img_known = cv2.imread("data/enrollment/922524243069/sample_01.jpg")
    # 2. Load another registered student 922524243074 sample
    img_other_known = cv2.imread("data/enrollment/922524243074/sample_01.jpg")

    from core.scrfd_detector import SCRFDDetector
    det = SCRFDDetector()
    aligner = FaceAligner()
    assessor = FaceQualityAssessor()

    # Test 1: Known student 1
    d1 = det.detect(img_known)[0]
    q1 = assessor.assess(d1, img_known, "stu1")
    t1 = aligner.align_and_normalize(img_known, d1["keypoints"])
    r1 = recognizer.recognize_face(RecognitionReadyFace(quality_metrics=q1, original_bbox=d1["bbox"], aligned_face_tensor=t1, timestamp=0.0))
    print(f"TEST Known Student 1: Status={r1.status} | Matched={r1.matched_student_id} ({r1.matched_student_name}) | Sim={r1.similarity:.3f} | Margin={r1.margin:.3f}")

    # Test 2: Known student 2
    d2 = det.detect(img_other_known)[0]
    q2 = assessor.assess(d2, img_other_known, "stu2")
    t2 = aligner.align_and_normalize(img_other_known, d2["keypoints"])
    r2 = recognizer.recognize_face(RecognitionReadyFace(quality_metrics=q2, original_bbox=d2["bbox"], aligned_face_tensor=t2, timestamp=0.0))
    print(f"TEST Known Student 2: Status={r2.status} | Matched={r2.matched_student_id} ({r2.matched_student_name}) | Sim={r2.similarity:.3f} | Margin={r2.margin:.3f}")

    # Test 3: Synthetic impostor / unknown student
    emb_unknown = np.random.randn(1, 3, 112, 112).astype(np.float32)
    q3 = FaceQualityResult(face_id="unk", bbox=[0,0,50,50], width=50, height=50, area=2500, sharpness=80, brightness=100, quality_status="RECOGNITION_READY")
    r3 = recognizer.recognize_face(RecognitionReadyFace(quality_metrics=q3, original_bbox=[0,0,50,50], aligned_face_tensor=emb_unknown, timestamp=0.0))
    print(f"TEST Impostor / Unknown: Status={r3.status} | Matched={r3.matched_student_id} | Sim={r3.similarity:.3f} | Margin={r3.margin:.3f}")

    # Test 4: Similar-looking ambiguity / narrow margin test
    # If best sim is 0.52 and second best is 0.49 (margin = 0.03 < 0.08), should reject as UNKNOWN!
    dummy_query = np.zeros(512, dtype=np.float32)
    # Search top 2 in vector store
    top_matches = vector_store.search(dummy_query, top_k=2)
    print("Vector store searchable with top-k:", len(top_matches))

if __name__ == "__main__":
    test_impostor_rejection()
