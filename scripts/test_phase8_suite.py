import os
import sys
import time
import datetime
import tempfile
import psutil
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from attendance.session_manager import SessionManager
from attendance.attendance_engine import AttendanceEngine
from core.tracking_pipeline import TrackingPipeline
from core.byte_tracker import ByteTracker
from core.recognizer import FaceRecognizer
from core.temporal_stabilizer import TemporalStabilizer
from core.schemas import TrackedFace, TrackState, RecognitionStatus, AttendanceEvent, AttendanceStatus
from scripts.test_phase7_suite import create_synthetic_classroom_frame


def run_phase8_benchmark():
    print("=" * 70)
    print("SMARTCLASS VISION AI - PHASE 8 BENCHMARK & PERFORMANCE SUITE")
    print("=" * 70)

    process = psutil.Process(os.getpid())
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "bench_smartclass_p8.sqlite")
    db = DatabaseManager(db_path=db_path)
    sm = SessionManager(db_manager=db)
    engine = AttendanceEngine(db_manager=db, session_manager=sm)

    # 1. Register 60 synthetic students
    print("\n[1/5] Populating 60 Synthetic Student Profiles...")
    t0 = time.perf_counter()
    for i in range(1, 61):
        stu_id = f"STU{i:03d}"
        stu_name = f"Test Student {i}"
        db.add_student(stu_id, stu_name, "Computer Science", "AIDS-B")
    reg_latency = (time.perf_counter() - t0) * 1000.0
    print(f"  Enrolled 60 students in {reg_latency:.2f} ms ({reg_latency / 60.0:.3f} ms / student)")

    # 2. Session Operation Latency
    print("\n[2/5] Measuring Session Operation Latencies...")
    t0 = time.perf_counter()
    sess = sm.create_session(
        session_id="BENCH_SESS_01",
        date=datetime.date.today().isoformat(),
        class_section="AIDS-B",
        subject="Deep Learning",
        planned_start_time="09:15:00",
        planned_end_time="10:00:00"
    )
    create_lat = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    sm.start_session("BENCH_SESS_01")
    start_lat = (time.perf_counter() - t0) * 1000.0

    print(f"  Session Create Latency: {create_lat:.3f} ms")
    print(f"  Session Start Latency:  {start_lat:.3f} ms")

    # 3. Database Write & Attendance Ingestion Latencies
    print("\n[3/5] Measuring Attendance Event Ingestion & DB Write Latencies...")
    event_latencies = []
    db_write_latencies = []

    for i in range(1, 61):
        stu_id = f"STU{i:03d}"
        event = AttendanceEvent(
            session_id="BENCH_SESS_01",
            student_id=stu_id,
            student_name=f"Test Student {i}",
            track_id=i,
            timestamp=time.time(),
            recognition_status=RecognitionStatus.MATCH,
            similarity=0.88,
            quality_status="RECOGNITION_READY"
        )
        t_evt = time.perf_counter()
        _ = engine.process_attendance_event(event)
        event_latencies.append((time.perf_counter() - t_evt) * 1000.0)

    avg_event_lat = float(np.mean(event_latencies))
    p95_event_lat = float(np.percentile(event_latencies, 95))

    print(f"  Single Event Ingestion Latency (Mean): {avg_event_lat:.3f} ms ({1000.0 / avg_event_lat:.1f} events/sec)")
    print(f"  Single Event Ingestion Latency (P95):  {p95_event_lat:.3f} ms")

    # Deduplicated Event Ingestion Latency (In-Memory Filtering)
    engine.clear_cache()
    # First write sets cache
    _ = engine.process_attendance_event(event)
    dedup_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = engine.process_attendance_event(event)
        dedup_times.append((time.perf_counter() - t0) * 1000.0)
    avg_dedup_lat = float(np.mean(dedup_times))
    print(f"  Deduplicated In-Memory Latency (Mean): {avg_dedup_lat:.4f} ms ({1000.0 / avg_dedup_lat:.1f} frames/sec)")

    # 4. End-to-End Tracking Pipeline Integration
    print("\n[4/5] Testing End-to-End Pipeline Integration (Tracking -> Attendance)...")
    pipeline = TrackingPipeline()
    tracked_faces_history = []

    # Run 10 consecutive frames with 12 students in the active session
    pipe_latencies = []
    for f in range(1, 11):
        frame, dets = create_synthetic_classroom_frame(1920, 1080, num_faces=12, frame_idx=f)
        t_pipe = time.perf_counter()
        tracked = pipeline.process_frame(frame, detections=dets)
        # Manually attach synthetic stable student IDs to simulate confirmed Phase 7 tracks
        for t in tracked:
            t.stable_student_id = f"STU{t.track_id:03d}"
            t.current_status = RecognitionStatus.MATCH
            t.current_similarity = 0.86

        # Feed directly to AttendanceEngine
        records = engine.process_tracked_faces("BENCH_SESS_01", tracked)
        pipe_latencies.append((time.perf_counter() - t_pipe) * 1000.0)

    avg_pipe_lat = float(np.mean(pipe_latencies))
    print(f"  End-to-End Frame Latency (Track + Attend): {avg_pipe_lat:.2f} ms ({1000.0 / avg_pipe_lat:.1f} FPS)")

    # 5. Session Report Generation & Final Stats
    print("\n[5/5] Generating Final Session Attendance Report...")
    t0 = time.perf_counter()
    report = engine.generate_session_report("BENCH_SESS_01")
    report_lat = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    sm.end_session("BENCH_SESS_01")
    end_lat = (time.perf_counter() - t0) * 1000.0

    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_pct = psutil.cpu_percent(interval=0.1)

    print(f"  Session End Latency:    {end_lat:.3f} ms")
    print(f"  Report Gen Latency:     {report_lat:.3f} ms")
    print(f"  Total Enrolled:         {report.total_enrolled}")
    print(f"  Present Count:          {report.present_count}")
    print(f"  Late Count:             {report.late_count}")
    print(f"  Not Seen Count:         {report.not_seen_count}")
    print(f"  Process RAM Usage:      {mem_mb:.1f} MB")
    print(f"  Process CPU Usage:      {cpu_pct:.1f} %")
    print("=" * 70)
    print("PHASE 8 BENCHMARK SUITE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_phase8_benchmark()
