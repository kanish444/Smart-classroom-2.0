import os
import sys
import time
import tempfile
import cv2
import numpy as np
import psutil
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.face_embedder import ArcFaceEmbedder
from database.db_manager import DatabaseManager
from core.vector_store import FaissVectorStore
from core.recognizer import FaceRecognizer
from core.calibration import ThresholdCalibrator
from core.face_processor import FaceProcessor
from enrollment.enrollment_service import EnrollmentService
from core.schemas import RecognitionReadyFace, FaceQualityResult, RecognitionStatus
from config.settings import get_settings


def create_test_face_tensor(seed: int = 42) -> np.ndarray:
    """Generates a deterministic synthetic face tensor (1, 3, 112, 112)."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-1.0, 1.0, size=(1, 3, 112, 112)).astype(np.float32)


def run_phase6_benchmark():
    logger.info("=" * 65)
    logger.info("SMARTCLASS VISION AI — PHASE 6 COMPLETE VALIDATION SUITE")
    logger.info("=" * 65)

    process = psutil.Process(os.getpid())
    cpu_before = psutil.cpu_percent(interval=0.1)
    ram_before_mb = process.memory_info().rss / (1024 * 1024)

    # 1. Model Loading
    logger.info("[1/10] Loading ArcFace MobileFaceNet Model...")
    t0 = time.perf_counter()
    embedder = ArcFaceEmbedder()
    load_time_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(f" -> ArcFace Loaded in {load_time_ms:.2f} ms | Output Dim: {embedder.embedding_dim}")

    # 2. Embedding Generation & Validation
    logger.info("[2/10] Validating Embedding Generation & Normalization...")
    t0 = time.perf_counter()
    sample_tensor = create_test_face_tensor(seed=123)
    emb = embedder.generate_embedding(sample_tensor)
    emb_gen_time_ms = (time.perf_counter() - t0) * 1000.0

    emb_norm = float(np.linalg.norm(emb))
    is_valid = embedder.validate_embedding(emb)
    logger.info(
        f" -> Shape: {emb.shape} | Dtype: {emb.dtype} | L2 Norm: {emb_norm:.6f} | "
        f"Validation: {'PASS' if is_valid else 'FAIL'} | Latency: {emb_gen_time_ms:.2f} ms"
    )

    # Setup isolated test environment
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "bench_db.sqlite")
        index_path = os.path.join(tmpdir, "bench_faiss.bin")
        db = DatabaseManager(db_path=db_path)
        store = FaissVectorStore(embedding_dim=512, index_path=index_path)
        enrollment = EnrollmentService(db, store, embedder)
        recognizer = FaceRecognizer(embedder=embedder, vector_store=store, threshold=0.85)

        # 3. Authorized Enrollment of Test Identities
        logger.info("[3/10] Enrolling Authorized Test Identities (Multi-sample)...")
        # Enrolling 5 test identities, each with 3 varied sample tensors
        test_students = [
            ("TEST_01", "Test Student Alpha"),
            ("TEST_02", "Test Student Beta"),
            ("TEST_03", "Test Student Gamma"),
            ("TEST_04", "Test Student Delta"),
            ("TEST_05", "Test Student Epsilon"),
        ]

        total_enrolled = 0
        for s_idx, (s_id, s_name) in enumerate(test_students):
            enrollment.register_student(s_id, s_name)
            for angle in ["frontal", "slight_left", "slight_right"]:
                # Deterministic seed per student & angle
                seed = (s_idx + 1) * 100 + (0 if angle == "frontal" else (1 if angle == "slight_left" else 2))
                tensor = create_test_face_tensor(seed=seed)
                success, emb_id, _ = enrollment.enroll_face_sample(
                    student_id=s_id,
                    face_tensor=tensor,
                    sample_label=angle,
                    quality_score=0.95
                )
                if success:
                    total_enrolled += 1

        logger.info(
            f" -> Enrolled {len(test_students)} Test Identities with {total_enrolled} Samples. "
            f"FAISS Total: {store.total_vectors} vectors."
        )

        # 4. Extract Real Faces from stress_test_result.jpg for Recognition Benchmarks
        logger.info("[4/10] Extracting Test Faces from stress_test_result.jpg...")
        real_img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stress_test_result.jpg")
        
        from core.detector import YOLOv8FaceDetector
        detector = YOLOv8FaceDetector(model_path="yolov8n.pt")
        processor = FaceProcessor()
        real_img = cv2.imread(real_img_path)
        dets = detector.detect(real_img)
        real_detections = [d for d in dets if d.get('class_id', 0) == 0]
        ready_faces, _ = processor.process(real_detections, real_img)
        logger.info(f" -> Extracted {len(ready_faces)} Phase 5 recognition-ready face tensors from real 1080p image.")

        # Enroll Subject 1 (face 0) and Subject 2 (face 1)
        enrollment.register_student("STUD_01", "Test Student Alpha")
        enrollment.register_student("STUD_02", "Test Student Beta")
        enrollment.enroll_face_sample("STUD_01", ready_faces[0].aligned_face_tensor, sample_label="frontal")
        enrollment.enroll_face_sample("STUD_02", ready_faces[1].aligned_face_tensor, sample_label="frontal")

        # Set default production threshold: 0.65
        recognizer.set_threshold(0.65)

        # 5. Correct Identity Matching Verification
        logger.info("[5/10] Verifying Correct Identity Matching...")
        res_match = recognizer.recognize_face(ready_faces[0])
        logger.info(
            f" -> Query Subject 1 -> Matched: {res_match.matched_student_id} ({res_match.matched_student_name}) | "
            f"Status: {res_match.status} | Sim: {res_match.similarity:.4f} | Latency: {res_match.processing_time_ms:.2f} ms"
        )
        match_pass = (res_match.status == RecognitionStatus.MATCH and res_match.matched_student_id == "STUD_01")

        # 6. Unknown Rejection Verification
        logger.info("[6/10] Verifying Unknown Rejection on Unenrolled Test Face...")
        # Face 2 is un-enrolled
        res_unk = recognizer.recognize_face(ready_faces[2])
        logger.info(
            f" -> Query Unenrolled (Face 2) -> Matched: {res_unk.matched_student_id} | "
            f"Status: {res_unk.status} | Top Sim: {res_unk.similarity:.4f} (Threshold: {recognizer.threshold}) | "
            f"Latency: {res_unk.processing_time_ms:.2f} ms"
        )
        unknown_pass = (res_unk.status == RecognitionStatus.UNKNOWN and res_unk.matched_student_id is None)

        # 7. Multi-Face Independent Recognition Batch
        logger.info("[7/10] Verifying Multi-Face Independent Batch Recognition...")
        # Batch: [Face 0 (Subject 1), Face 2 (Unenrolled), Face 1 (Subject 2), Face 3 (Unenrolled), Face 4 (Unenrolled)]
        batch_res = recognizer.recognize_batch(ready_faces)
        for idx, r in enumerate(batch_res):
            logger.info(f"    Face {idx}: Status={r.status}, MatchID={r.matched_student_id}, Sim={r.similarity:.4f}, Latency={r.processing_time_ms:.2f} ms")

        multi_pass = (
            batch_res[0].status == RecognitionStatus.MATCH and batch_res[0].matched_student_id == "STUD_01" and
            batch_res[1].status == RecognitionStatus.MATCH and batch_res[1].matched_student_id == "STUD_02" and
            batch_res[2].status == RecognitionStatus.UNKNOWN and
            batch_res[3].status == RecognitionStatus.UNKNOWN and
            batch_res[4].status == RecognitionStatus.UNKNOWN
        )

        # 8. Threshold Calibration & Distribution Infrastructure
        logger.info("[8/10] Evaluating Threshold Calibration Infrastructure on Real Test Faces...")
        # Compute embeddings for all 5 faces
        embs_all = [embedder.generate_embedding(rf.aligned_face_tensor) for rf in ready_faces]
        
        # Genuine pairs (identical face or slight perturbation)
        genuine_scores = [float(np.dot(embs_all[0], embs_all[0].T)[0, 0])]
        # Impostor pairs (all distinct faces against each other)
        impostor_scores = []
        for i in range(len(embs_all)):
            for j in range(i + 1, len(embs_all)):
                imp_sim = float(np.dot(embs_all[i], embs_all[j].T)[0, 0])
                impostor_scores.append(imp_sim)

        calib_res = ThresholdCalibrator.evaluate_distributions(
            genuine_similarities=genuine_scores,
            impostor_similarities=impostor_scores,
            threshold=recognizer.threshold
        )
        logger.info(f" -> Calibration Status: {calib_res['calibration_status']}")
        if calib_res['calibration_status'] == "CALIBRATION_COMPUTED":
            g_dist = calib_res['genuine_distribution']
            i_dist = calib_res['impostor_distribution']
            metrics = calib_res['metrics']
            logger.info(f"    Genuine Pairs ({g_dist['count']}): Mean Sim = {g_dist['mean']:.4f} (Min={g_dist['min']:.4f}, Max={g_dist['max']:.4f})")
            logger.info(f"    Impostor Pairs ({i_dist['count']}): Mean Sim = {i_dist['mean']:.4f} (Min={i_dist['min']:.4f}, Max={i_dist['max']:.4f})")
            logger.info(f"    Evaluation: FAR={metrics['FAR']:.4f}, FRR={metrics['FRR']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        else:
            logger.info(f"    {calib_res.get('message')}")

        # 9. Systematic Latency & Throughput Benchmark
        logger.info("[9/10] Measuring Systematic Latency and Scalability Benchmarks...")
        # Benchmark FAISS standalone search latency
        dummy_vec = np.random.randn(1, 512).astype(np.float32)
        dummy_vec = dummy_vec / np.linalg.norm(dummy_vec)
        t0 = time.perf_counter()
        faiss_runs = 500
        for _ in range(faiss_runs):
            store.search(dummy_vec, top_k=1)
        faiss_search_latency_ms = ((time.perf_counter() - t0) / faiss_runs) * 1000.0
        logger.info(f" -> Standalone FAISS Search Latency: {faiss_search_latency_ms:.4f} ms/search ({1000.0/faiss_search_latency_ms:.0f} searches/sec)")

        # Benchmark standalone embedding extraction latency
        t0 = time.perf_counter()
        emb_runs = 50
        for _ in range(emb_runs):
            embedder.generate_embedding(sample_tensor)
        avg_emb_latency_ms = ((time.perf_counter() - t0) / emb_runs) * 1000.0
        logger.info(f" -> Average ArcFace Embedding Latency: {avg_emb_latency_ms:.2f} ms/face")

        # Scalability: 1 face, 5 faces, 10 faces, 20 faces
        scalability_results = {}
        for count in [1, 5, 10, 20]:
            test_batch = [ready_faces[0]] * count
            t0 = time.perf_counter()
            runs = 20
            for _ in range(runs):
                recognizer.recognize_batch(test_batch)
            elapsed = time.perf_counter() - t0
            lat_batch_ms = (elapsed / runs) * 1000.0
            fps = (count * runs) / elapsed
            scalability_results[count] = {"batch_lat_ms": lat_batch_ms, "fps": fps}
            logger.info(f" -> {count} Faces: Total Batch = {lat_batch_ms:.2f} ms ({lat_batch_ms/count:.2f} ms/face) | Throughput: {fps:.1f} faces/sec")

        # 10. Hardware Resource Usage
        logger.info("[10/10] Resource Utilization Measurements...")
        cpu_after = psutil.cpu_percent(interval=0.2)
        ram_after_mb = process.memory_info().rss / (1024 * 1024)
        has_gpu = False
        gpu_info = "N/A (CPU execution provider utilized)"
        try:
            import torch
            if torch.cuda.is_available():
                has_gpu = True
                gpu_info = torch.cuda.get_device_name(0)
        except Exception:
            pass

        logger.info(f" -> CPU Usage: {cpu_after:.1f}%")
        logger.info(f" -> RAM Usage: {ram_after_mb:.1f} MB (Delta: +{ram_after_mb - ram_before_mb:.1f} MB)")
        logger.info(f" -> GPU: {gpu_info}")

        logger.info("=" * 65)
        logger.info("PHASE 6 VALIDATION SUMMARY:")
        logger.info(f"Model Loading Time       : {load_time_ms:.2f} ms")
        logger.info(f"Embedding Latency (avg)  : {avg_emb_latency_ms:.2f} ms/face")
        logger.info(f"FAISS Search Latency     : {faiss_search_latency_ms:.4f} ms")
        logger.info(f"1-Face Recognition Time  : {scalability_results[1]['batch_lat_ms']:.2f} ms")
        logger.info(f"10-Face Recognition Time : {scalability_results[10]['batch_lat_ms']:.2f} ms")
        logger.info(f"20-Face Recognition Time : {scalability_results[20]['batch_lat_ms']:.2f} ms")
        logger.info(f"Identity Match Test      : {'PASS' if match_pass else 'FAIL'}")
        logger.info(f"Unknown Rejection Test   : {'PASS' if unknown_pass else 'FAIL'}")
        logger.info(f"Multi-face Test          : {'PASS' if multi_pass else 'FAIL'}")
        logger.info("=" * 65)

        return {
            "load_time_ms": load_time_ms,
            "avg_emb_latency_ms": avg_emb_latency_ms,
            "faiss_search_latency_ms": faiss_search_latency_ms,
            "scalability": scalability_results,
            "match_pass": match_pass,
            "unknown_pass": unknown_pass,
            "multi_pass": multi_pass,
            "cpu_usage": cpu_after,
            "ram_usage_mb": ram_after_mb,
            "gpu_info": gpu_info,
            "calib_res": calib_res
        }


if __name__ == "__main__":
    run_phase6_benchmark()
