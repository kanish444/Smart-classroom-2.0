import os
import time
import datetime
import tempfile
import sqlite3
import pytest

from database.db_manager import DatabaseManager
from database.new_enrollment_db import NewEnrollmentDatabase
from attendance.session_manager import SessionManager
from attendance.attendance_engine import AttendanceEngine
from core.schemas import TrackedFace, TrackState, RecognitionStatus, AttendanceEvent, AttendanceStatus


@pytest.fixture
def isolated_attendance_env(tmp_path):
    """Provides fully isolated new_enrollment and smartclass SQLite databases."""
    new_db_file = str(tmp_path / "new_enrollment.sqlite")
    main_db_file = str(tmp_path / "smartclass.sqlite")

    new_db = NewEnrollmentDatabase(db_path=new_db_file)
    db_mgr = DatabaseManager(db_path=main_db_file, new_db_path=new_db_file)
    sess_mgr = SessionManager(db_manager=db_mgr)
    engine = AttendanceEngine(db_manager=db_mgr, session_manager=sess_mgr)

    return {
        "new_db": new_db,
        "new_db_file": new_db_file,
        "db": db_mgr,
        "sess_mgr": sess_mgr,
        "engine": engine,
        "tmp_path": tmp_path
    }


def _seed_student(new_db: NewEnrollmentDatabase, db_mgr: DatabaseManager, reg_no: str, name: str, dept: str = "AI&DS", sec: str = "B"):
    import numpy as np
    dummy_vec = np.zeros(512, dtype=np.float32)
    new_db.save_enrollment_atomic(
        student_id=reg_no,
        register_number=reg_no,
        name=name,
        class_name="3rd Year",
        department=dept,
        section=sec,
        embeddings=[dummy_vec]
    )
    db_mgr.add_student(
        student_id=reg_no,
        student_name=name,
        department=dept,
        section=sec,
        status="active",
        register_no=reg_no,
        class_name=f"{dept} - {sec}"
    )


def test_1_zero_enrolled_students(isolated_attendance_env):
    """TEST 1: 0 enrolled students -> attendance roster = 0."""
    env = isolated_attendance_env
    sess = env["sess_mgr"].create_session("SESS_T1", "2026-09-25", "ALL", "AI Lab", "09:00", "11:00")
    report = env["engine"].generate_session_report(sess.session_id)

    assert report.total_enrolled == 0
    assert len(report.records) == 0
    assert len(report.not_seen_students) == 0
    assert report.present_count == 0


def test_2_one_enrolled_student_recognition(isolated_attendance_env):
    """
    TEST 2: 1 enrolled student -> roster = 1 -> initially PRESENT = 0
    -> after confirmed recognition PRESENT = 1.
    """
    env = isolated_attendance_env

    _seed_student(env["new_db"], env["db"], "922524243080", "KAVIRAJ")

    sess = env["sess_mgr"].create_session("SESS_T2", "2026-09-25", "ALL", "AI Lab", "09:00", "11:00")
    sess_active = env["sess_mgr"].start_session(sess.session_id)

    # Initial roster: total=1, present=0, not_seen=1
    init_report = env["engine"].generate_session_report(sess.session_id)
    assert init_report.total_enrolled == 1
    assert init_report.present_count == 0
    assert init_report.not_seen_count == 1
    assert init_report.not_seen_students == ["922524243080"]

    # Face recognized and stabilized
    face = TrackedFace(
        track_id=1,
        bbox=[100, 100, 200, 200],
        state=TrackState.ACTIVE,
        score=0.95,
        stable_student_id="922524243080",
        stable_student_name="KAVIRAJ",
        current_status=RecognitionStatus.MATCH,
        current_similarity=0.88,
        quality_status="RECOGNITION_READY"
    )

    records = env["engine"].process_tracked_faces(sess.session_id, [face])
    assert len(records) == 1
    assert records[0].status == AttendanceStatus.PRESENT

    # After recognition report
    final_report = env["engine"].generate_session_report(sess.session_id)
    assert final_report.total_enrolled == 1
    assert final_report.present_count == 1
    assert final_report.not_seen_count == 0
    assert len(final_report.records) == 1
    assert final_report.records[0].student_id == "922524243080"
    assert final_report.records[0].student_name == "KAVIRAJ"


def test_3_two_enrolled_students_recognition(isolated_attendance_env):
    """
    TEST 3: 2 enrolled students -> roster = 2 -> initially PRESENT = 0
    -> after both confirmed recognition PRESENT = 2.
    """
    env = isolated_attendance_env

    _seed_student(env["new_db"], env["db"], "922524243080", "KAVIRAJ")
    _seed_student(env["new_db"], env["db"], "922524243074", "KARTHICK RAJA")

    sess = env["sess_mgr"].create_session("SESS_T3", "2026-09-25", "ALL", "AI Lab", "09:00", "11:00")
    env["sess_mgr"].start_session(sess.session_id)

    # Initial state
    init_report = env["engine"].generate_session_report(sess.session_id)
    assert init_report.total_enrolled == 2
    assert init_report.present_count == 0
    assert init_report.not_seen_count == 2

    # Recognize both students
    face1 = TrackedFace(
        track_id=1,
        bbox=[100, 100, 200, 200],
        state=TrackState.ACTIVE,
        score=0.95,
        stable_student_id="922524243080",
        stable_student_name="KAVIRAJ",
        current_status=RecognitionStatus.MATCH,
        current_similarity=0.89,
        quality_status="RECOGNITION_READY"
    )
    face2 = TrackedFace(
        track_id=2,
        bbox=[300, 100, 400, 200],
        state=TrackState.ACTIVE,
        score=0.92,
        stable_student_id="922524243074",
        stable_student_name="KARTHICK RAJA",
        current_status=RecognitionStatus.MATCH,
        current_similarity=0.85,
        quality_status="RECOGNITION_READY"
    )

    env["engine"].process_tracked_faces(sess.session_id, [face1, face2])

    report = env["engine"].generate_session_report(sess.session_id)
    assert report.total_enrolled == 2
    assert report.present_count == 2
    assert report.not_seen_count == 0
    assert len(report.records) == 2


def test_4_unknown_face_does_not_affect_roster(isolated_attendance_env):
    """TEST 4: UNKNOWN face -> roster count unchanged, PRESENT count unchanged."""
    env = isolated_attendance_env

    _seed_student(env["new_db"], env["db"], "922524243080", "KAVIRAJ")

    sess = env["sess_mgr"].create_session("SESS_T4", "2026-09-25", "ALL", "AI Lab", "09:00", "11:00")
    env["sess_mgr"].start_session(sess.session_id)

    # UNKNOWN face observed
    unknown_face = TrackedFace(
        track_id=99,
        bbox=[50, 50, 120, 120],
        state=TrackState.ACTIVE,
        score=0.88,
        stable_student_id=None,
        stable_student_name=None,
        current_status=RecognitionStatus.UNKNOWN,
        current_similarity=0.0,
        quality_status="RECOGNITION_READY"
    )

    recs = env["engine"].process_tracked_faces(sess.session_id, [unknown_face])
    assert len(recs) == 0

    report = env["engine"].generate_session_report(sess.session_id)
    assert report.total_enrolled == 1
    assert report.present_count == 0
    assert report.not_seen_count == 1
    assert "922524243080" in report.not_seen_students
    assert len(report.records) == 0


def test_5_same_student_visible_100_frames(isolated_attendance_env):
    """TEST 5: Same student visible for 100 frames -> attendance count remains 1."""
    env = isolated_attendance_env

    _seed_student(env["new_db"], env["db"], "922524243080", "KAVIRAJ")

    sess = env["sess_mgr"].create_session("SESS_T5", "2026-09-25", "ALL", "AI Lab", "09:00", "11:00")
    env["sess_mgr"].start_session(sess.session_id)

    face = TrackedFace(
        track_id=1,
        bbox=[100, 100, 200, 200],
        state=TrackState.ACTIVE,
        score=0.95,
        stable_student_id="922524243080",
        stable_student_name="KAVIRAJ",
        current_status=RecognitionStatus.MATCH,
        current_similarity=0.91,
        quality_status="RECOGNITION_READY"
    )

    base_time = time.time()
    for i in range(100):
        env["engine"].process_tracked_faces(sess.session_id, [face], timestamp=base_time + (i * 0.033))

    report = env["engine"].generate_session_report(sess.session_id)
    assert report.total_enrolled == 1
    assert report.present_count == 1
    assert report.not_seen_count == 0
    assert len(report.records) == 1, "Must contain exactly 1 attendance record for the student across 100 frames."


def test_6_stale_records_not_in_roster(isolated_attendance_env):
    """TEST 6: Old/test STU_* record not enrolled in current system -> must NOT appear in roster."""
    env = isolated_attendance_env

    # Legitimate student
    _seed_student(env["new_db"], env["db"], "922524243080", "KAVIRAJ")

    # Attempt to inject stale test student directly into legacy students table
    with env["db"].get_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO students (student_id, student_name, department, section) VALUES ('STU_DTEST_9999', 'Fake Test Student', 'TestDept', 'A');")
        conn.commit()

    # Sync cleans up stale records
    env["db"].sync_enrolled_students()

    sess = env["sess_mgr"].create_session("SESS_T6", "2026-09-25", "ALL", "AI Lab", "09:00", "11:00")
    report = env["engine"].generate_session_report(sess.session_id)

    # Stale student must NOT be in roster
    assert "STU_DTEST_9999" not in report.not_seen_students
    assert all(r.student_id != "STU_DTEST_9999" for r in report.records)
    assert report.total_enrolled == 1


def test_7_recognition_points_to_valid_enrollment(isolated_attendance_env):
    """TEST 7: Recognition result points to valid enrollment -> correct register number/name marked PRESENT."""
    env = isolated_attendance_env

    _seed_student(env["new_db"], env["db"], "922524243074", "KARTHICK RAJA")

    sess = env["sess_mgr"].create_session("SESS_T7", "2026-09-25", "ALL", "AI Lab", "09:00", "11:00")
    env["sess_mgr"].start_session(sess.session_id)

    # Recognition resolves to student_id="922524243074"
    face = TrackedFace(
        track_id=5,
        bbox=[150, 150, 250, 250],
        state=TrackState.ACTIVE,
        score=0.96,
        stable_student_id="922524243074",
        stable_student_name="KARTHICK RAJA",
        current_status=RecognitionStatus.MATCH,
        current_similarity=0.87,
        quality_status="RECOGNITION_READY"
    )

    env["engine"].process_tracked_faces(sess.session_id, [face])

    report = env["engine"].generate_session_report(sess.session_id)
    assert report.present_count == 1
    assert len(report.records) == 1
    assert report.records[0].student_id == "922524243074"
    assert report.records[0].student_name == "KARTHICK RAJA"
    assert report.records[0].status == AttendanceStatus.PRESENT
