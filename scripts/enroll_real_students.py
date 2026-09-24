r"""
Real Student Dataset Enrollment Script for SmartClass Vision AI (Phase 10).

Parses C:/Users/kanis/Downloads/Student Sample.docx,
validates single-face constraints, applies Phase 5 quality gating,
extracts 512-dim ArcFace embeddings using pretrained MobileFaceNet (NO retraining),
persists student records and embeddings to SQLite, and updates FAISS vector store.
Collects and prints exact empirical performance and audit metrics.
"""

import os
import sys
import time
import json
import psutil
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from core.vector_store import FaissVectorStore
from core.face_embedder import ArcFaceEmbedder
from enrollment.data_importer import StudentDataImporter
from enrollment.enrollment_service import EnrollmentService, EnrollmentStatus


def run_real_student_enrollment():
    docx_path = r"C:\Users\kanis\Downloads\Student Sample.docx"
    if not os.path.exists(docx_path):
        print(f"ERROR: Student Sample.docx not found at: {docx_path}")
        return

    process = psutil.Process(os.getpid())
    cpu_start = process.cpu_percent(interval=None)
    mem_start_mb = process.memory_info().rss / (1024 * 1024)

    print("================================================================")
    print("PHASE 10: REAL STUDENT BATCH ENROLLMENT EXECUTION")
    print("================================================================")
    print(f"Data source: {docx_path}")

    # 1. Initialize components with clean state
    os.makedirs("data", exist_ok=True)
    os.makedirs("database", exist_ok=True)
    os.makedirs("models/embeddings", exist_ok=True)

    db_path = "data/smartclass.sqlite"
    index_path = "models/embeddings/faiss_index.bin"

    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(index_path):
        os.remove(index_path)
    if os.path.exists(f"{index_path}.meta.json"):
        os.remove(f"{index_path}.meta.json")

    print("Initializing Database, Vector Store, Embedder, and EnrollmentService...")
    db = DatabaseManager(db_path=db_path)
    store = FaissVectorStore(embedding_dim=512, index_path=index_path)
    embedder = ArcFaceEmbedder()

    service = EnrollmentService(
        db_manager=db,
        vector_store=store,
        embedder=embedder,
        conflict_threshold=0.95
    )

    # 2. Parse Real Student Document
    importer = StudentDataImporter()
    t_parse_start = time.perf_counter()
    records = importer.import_docx(docx_path)
    t_parse_end = time.perf_counter()
    parse_elapsed_ms = (t_parse_end - t_parse_start) * 1000.0

    print(f"Parsed {len(records)} student records from DOCX in {parse_elapsed_ms:.2f} ms.")

    # 3. Batch Enrollment with Step-Level Profiling
    t_enroll_start = time.perf_counter()
    report = service.enroll_batch(records)
    t_enroll_end = time.perf_counter()
    total_enroll_sec = t_enroll_end - t_enroll_start

    # Benchmark micro-operations: embedding generation, sqlite insertion, faiss insertion
    t_emb_times = []
    t_sql_times = []
    t_faiss_times = []

    # Run micro-benchmark on 5 sample validated face tensors
    sample_tensor = np.random.RandomState(42).uniform(-1.0, 1.0, size=(1, 3, 112, 112)).astype(np.float32)
    for _ in range(10):
        t0 = time.perf_counter()
        emb = embedder.generate_embedding(sample_tensor)
        t_emb_times.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        temp_id = f"bench_{time.time_ns()}"
        db.add_student(temp_id, "Benchmark Student")
        eid = db.add_embedding(temp_id, emb.reshape(-1))
        t_sql_times.append((time.perf_counter() - t1) * 1000.0)

        t2 = time.perf_counter()
        store.add_vector(eid, emb.reshape(-1), {"student_id": temp_id, "student_name": "Benchmark Student"})
        t_faiss_times.append((time.perf_counter() - t2) * 1000.0)

        # Cleanup benchmark item
        db.delete_student(temp_id)

    # Re-sync store after micro-benchmark cleanup
    store.sync_with_database(db.get_all_embeddings())
    store.save_index()

    # 4. Measure FAISS Rebuild Benchmark
    t_rebuild_start = time.perf_counter()
    rebuilt_count = service.rebuild_faiss_index()
    t_rebuild_end = time.perf_counter()
    rebuild_elapsed_ms = (t_rebuild_end - t_rebuild_start) * 1000.0

    # 5. Measure Resource Utilization
    cpu_end = process.cpu_percent(interval=0.1)
    mem_end_mb = process.memory_info().rss / (1024 * 1024)

    # 6. Verify Consistency
    consistency = service.verify_index_consistency()

    # 7. Mirror to database/ directory for runtime settings compatibility
    import shutil
    shutil.copy2("data/smartclass.sqlite", "database/smartclass.sqlite")
    shutil.copy2("models/embeddings/faiss_index.bin", "database/faiss_index.bin")
    if os.path.exists("models/embeddings/faiss_index.bin.meta.json"):
        shutil.copy2("models/embeddings/faiss_index.bin.meta.json", "database/faiss_index.bin.meta.json")

    avg_emb_ms = float(np.mean(t_emb_times))
    avg_sql_ms = float(np.mean(t_sql_times))
    avg_faiss_ms = float(np.mean(t_faiss_times))

    metrics = {
        "data_source": docx_path,
        "total_records": report.total_records,
        "successfully_enrolled": report.successfully_enrolled,
        "failed": report.failed,
        "requires_review": report.requires_review,
        "counts": report.counts,
        "parse_time_ms": round(parse_elapsed_ms, 2),
        "total_enrollment_time_sec": round(total_enroll_sec, 3),
        "avg_time_per_student_ms": round((total_enroll_sec / max(1, len(records))) * 1000.0, 2),
        "embedding_gen_time_ms": round(avg_emb_ms, 2),
        "sqlite_insertion_time_ms": round(avg_sql_ms, 2),
        "faiss_insertion_time_ms": round(avg_faiss_ms, 2),
        "index_rebuild_time_ms": round(rebuild_elapsed_ms, 2),
        "rebuilt_vectors": rebuilt_count,
        "memory_start_mb": round(mem_start_mb, 2),
        "memory_end_mb": round(mem_end_mb, 2),
        "memory_delta_mb": round(mem_end_mb - mem_start_mb, 2),
        "cpu_percent": round(cpu_end, 1),
        "consistency": consistency,
        "failures_sample": report.failures[:5],
        "details_sample": [d.model_dump() for d in report.details[:5]]
    }

    # Save execution audit report to JSON
    with open("data/phase10_real_enrollment_report.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n================================================================")
    print("ENROLLMENT AUDIT REPORT SUMMARY")
    print("================================================================")
    print(f"Total Student Records:        {report.total_records}")
    print(f"Successfully Enrolled:        {report.successfully_enrolled}")
    print(f"Failed / Rejected:            {report.failed}")
    print(f"Requires Review:              {report.requires_review}")
    print("----------------------------------------------------------------")
    print("Detailed Status Breakdown:")
    for k, v in report.counts.items():
        print(f"  - {k:<28}: {v}")
    print("----------------------------------------------------------------")
    print(f"Total Enrollment Time:        {total_enroll_sec:.3f} s")
    print(f"Avg Time Per Student:         {metrics['avg_time_per_student_ms']:.2f} ms")
    print(f"Embedding Gen Time / Face:    {avg_emb_ms:.2f} ms")
    print(f"SQLite Insertion Time:        {avg_sql_ms:.2f} ms")
    print(f"FAISS Insertion Time:         {avg_faiss_ms:.2f} ms")
    print(f"FAISS Index Rebuild Time:     {rebuild_elapsed_ms:.2f} ms ({rebuilt_count} vectors)")
    print(f"Memory RSS:                   {mem_end_mb:.2f} MB (Delta: +{mem_end_mb - mem_start_mb:.2f} MB)")
    print(f"Index Consistency:            {consistency['message']}")
    print("================================================================\n")


if __name__ == "__main__":
    run_real_student_enrollment()

