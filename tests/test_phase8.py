import os
import sys
import time
import datetime
import tempfile
import sqlite3
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from attendance.session_manager import (
    SessionManager,
    SessionManagerError,
    InvalidSessionTransitionError,
    DuplicateSessionError
)
from attendance.attendance_engine import AttendanceEngine
from core.schemas import (
    SessionState,
    AttendanceStatus,
    SessionInfo,
    AttendanceRecord,
    AttendanceEvent,
    TrackedFace,
    TrackState,
    RecognitionStatus
)


@pytest.fixture
def test_env():
    """Provides an isolated temporary SQLite database and managers for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_smartclass_p8.sqlite")
        db = DatabaseManager(db_path=db_path)
        # Register standard synthetic test students
        db.add_student("STU001", "Alice Smith", "Computer Science", "AIDS-B")
        db.add_student("STU002", "Bob Jones", "Computer Science", "AIDS-B")
        db.add_student("STU003", "Charlie Brown", "Computer Science", "AIDS-B")
        db.add_student("STU004", "Diana Prince", "Computer Science", "AIDS-B")
        db.add_student("STU005", "Evan Wright", "Computer Science", "AIDS-B")

        sm = SessionManager(db_manager=db)
        engine = AttendanceEngine(db_manager=db, session_manager=sm)
        yield db, sm, engine, tmpdir


# ---------------------------------------------------------------------------
# Test 1: Create Session
# ---------------------------------------------------------------------------
def test_01_create_session(test_env):
    db, sm, engine, _ = test_env
    sess = sm.create_session(
        session_id="SESS_01",
        date="2026-09-23",
        class_section="AIDS-B",
        subject="Computer Vision",
        planned_start_time="09:15:00",
        planned_end_time="10:00:00"
    )
    assert sess.session_id == "SESS_01"
    assert sess.status == SessionState.SCHEDULED
    assert sess.class_section == "AIDS-B"
    assert sess.subject == "Computer Vision"


# ---------------------------------------------------------------------------
# Test 2: Start Session
# ---------------------------------------------------------------------------
def test_02_start_session(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_02", "2026-09-23", "AIDS-B", "Math", "09:00:00", "09:45:00")
    sess = sm.start_session("SESS_02", actual_start_time="2026-09-23T09:00:05")
    assert sess.status == SessionState.ACTIVE
    assert sess.actual_start_time == "2026-09-23T09:00:05"


# ---------------------------------------------------------------------------
# Test 3: End Session
# ---------------------------------------------------------------------------
def test_03_end_session(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_03", "2026-09-23", "AIDS-B", "Physics", "10:00:00", "10:45:00")
    sm.start_session("SESS_03")
    sess = sm.end_session("SESS_03", actual_end_time="2026-09-23T10:45:10")
    assert sess.status == SessionState.ENDED
    assert sess.actual_end_time == "2026-09-23T10:45:10"


# ---------------------------------------------------------------------------
# Test 4: Cancel Session
# ---------------------------------------------------------------------------
def test_04_cancel_session(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_04", "2026-09-23", "AIDS-B", "Chemistry", "11:00:00", "11:45:00")
    sess = sm.cancel_session("SESS_04")
    assert sess.status == SessionState.CANCELLED


# ---------------------------------------------------------------------------
# Test 5: Invalid Transition (ENDED -> ACTIVE)
# ---------------------------------------------------------------------------
def test_05_invalid_session_transition_ended_to_active(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_05", "2026-09-23", "AIDS-B", "English", "12:00:00", "12:45:00")
    sm.start_session("SESS_05")
    sm.end_session("SESS_05")

    with pytest.raises(InvalidSessionTransitionError):
        sm.start_session("SESS_05")


# ---------------------------------------------------------------------------
# Test 6: Invalid Transition (CANCELLED -> ACTIVE)
# ---------------------------------------------------------------------------
def test_06_invalid_session_transition_cancelled_to_active(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_06", "2026-09-23", "AIDS-B", "Biology", "13:00:00", "13:45:00")
    sm.cancel_session("SESS_06")

    with pytest.raises(InvalidSessionTransitionError):
        sm.start_session("SESS_06")


# ---------------------------------------------------------------------------
# Test 7: Recognized Student Marked PRESENT
# ---------------------------------------------------------------------------
def test_07_recognized_student_marked_present(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_07", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_07", actual_start_time="2026-09-23T09:00:00")

    # Student recognized at 09:05 (within 10m threshold)
    dt = datetime.datetime(2026, 9, 23, 9, 5, 0)
    event = AttendanceEvent(
        session_id="SESS_07",
        student_id="STU001",
        student_name="Alice Smith",
        track_id=10,
        timestamp=dt.timestamp(),
        recognition_status=RecognitionStatus.MATCH,
        similarity=0.85
    )
    rec = engine.process_attendance_event(event)

    assert rec is not None
    assert rec.student_id == "STU001"
    assert rec.status == AttendanceStatus.PRESENT
    assert rec.first_track_id == 10
    assert rec.initial_similarity == 0.85


# ---------------------------------------------------------------------------
# Test 8: Same Student Recognized Repeatedly
# ---------------------------------------------------------------------------
def test_08_same_student_recognized_repeatedly(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_08", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_08", actual_start_time="2026-09-23T09:00:00")

    for minute in [2, 5, 10, 15, 20]:
        engine.clear_cache() # bypass frame dedup buffer
        dt = datetime.datetime(2026, 9, 23, 9, minute, 0)
        event = AttendanceEvent(
            session_id="SESS_08",
            student_id="STU001",
            track_id=12,
            timestamp=dt.timestamp(),
            recognition_status=RecognitionStatus.MATCH,
            similarity=0.80 + minute * 0.005
        )
        engine.process_attendance_event(event)

    records = db.get_attendance_for_session("SESS_08")
    assert len(records) == 1  # Exactly ONE record


# ---------------------------------------------------------------------------
# Test 9: Duplicate Attendance DB Prevention
# ---------------------------------------------------------------------------
def test_09_duplicate_attendance_db_prevention(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_09", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_09")

    # Direct raw insert test: violating UNIQUE (session_id, student_id)
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO attendance (session_id, student_id, status, first_seen, last_seen)
            VALUES ('SESS_09', 'STU001', 'PRESENT', '2026-09-23T09:01:00', '2026-09-23T09:01:00');
        """)
        conn.commit()

        # Second raw insert MUST raise IntegrityError
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("""
                INSERT INTO attendance (session_id, student_id, status, first_seen, last_seen)
                VALUES ('SESS_09', 'STU001', 'PRESENT', '2026-09-23T09:05:00', '2026-09-23T09:05:00');
            """)


# ---------------------------------------------------------------------------
# Test 10: First_seen Preservation
# ---------------------------------------------------------------------------
def test_10_first_seen_preservation(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_10", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_10", actual_start_time="2026-09-23T09:00:00")

    dt1 = datetime.datetime(2026, 9, 23, 9, 2, 15)
    event1 = AttendanceEvent(
        session_id="SESS_10", student_id="STU002", track_id=5,
        timestamp=dt1.timestamp(), recognition_status=RecognitionStatus.MATCH, similarity=0.88
    )
    engine.process_attendance_event(event1)
    rec1 = db.get_attendance_record("SESS_10", "STU002")
    initial_first_seen = rec1["first_seen"]

    engine.clear_cache()
    dt2 = datetime.datetime(2026, 9, 23, 9, 30, 45)
    event2 = AttendanceEvent(
        session_id="SESS_10", student_id="STU002", track_id=5,
        timestamp=dt2.timestamp(), recognition_status=RecognitionStatus.MATCH, similarity=0.91
    )
    engine.process_attendance_event(event2)
    rec2 = db.get_attendance_record("SESS_10", "STU002")

    # first_seen MUST remain identical to dt1
    assert rec2["first_seen"] == initial_first_seen
    assert rec2["first_seen"] == dt1.isoformat()


# ---------------------------------------------------------------------------
# Test 11: Last_seen Update
# ---------------------------------------------------------------------------
def test_11_last_seen_update(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_11", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_11")

    dt1 = datetime.datetime(2026, 9, 23, 9, 2, 0)
    dt2 = datetime.datetime(2026, 9, 23, 9, 25, 0)

    event1 = AttendanceEvent(session_id="SESS_11", student_id="STU003", track_id=8, timestamp=dt1.timestamp(), recognition_status=RecognitionStatus.MATCH, similarity=0.82)
    engine.process_attendance_event(event1)

    engine.clear_cache()
    event2 = AttendanceEvent(session_id="SESS_11", student_id="STU003", track_id=8, timestamp=dt2.timestamp(), recognition_status=RecognitionStatus.MATCH, similarity=0.89)
    engine.process_attendance_event(event2)

    rec = db.get_attendance_record("SESS_11", "STU003")
    assert rec["last_seen"] == dt2.isoformat()
    assert rec["latest_similarity"] == 0.89


# ---------------------------------------------------------------------------
# Test 12: UNKNOWN Recognition Ignored
# ---------------------------------------------------------------------------
def test_12_unknown_recognition_ignored(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_12", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_12")

    tf = TrackedFace(
        track_id=14,
        bbox=[100, 100, 200, 200],
        state=TrackState.ACTIVE,
        stable_student_id=None,  # UNKNOWN
        current_status=RecognitionStatus.UNKNOWN,
        current_similarity=0.42
    )
    records = engine.process_tracked_faces("SESS_12", [tf])
    assert len(records) == 0
    assert len(db.get_attendance_for_session("SESS_12")) == 0


# ---------------------------------------------------------------------------
# Test 13: INVALID Recognition Ignored
# ---------------------------------------------------------------------------
def test_13_invalid_recognition_ignored(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_13", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_13")

    tf = TrackedFace(
        track_id=15,
        bbox=[100, 100, 200, 200],
        state=TrackState.ACTIVE,
        stable_student_id="STU001",
        current_status=RecognitionStatus.INVALID,
        current_similarity=0.0
    )
    records = engine.process_tracked_faces("SESS_13", [tf])
    assert len(records) == 0


# ---------------------------------------------------------------------------
# Test 14: ERROR Recognition Ignored
# ---------------------------------------------------------------------------
def test_14_error_recognition_ignored(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_14", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_14")

    tf = TrackedFace(
        track_id=16,
        bbox=[100, 100, 200, 200],
        state=TrackState.ACTIVE,
        stable_student_id="STU001",
        current_status=RecognitionStatus.ERROR,
        current_similarity=0.0
    )
    records = engine.process_tracked_faces("SESS_14", [tf])
    assert len(records) == 0


# ---------------------------------------------------------------------------
# Test 15: Multiple Students Marked Independently
# ---------------------------------------------------------------------------
def test_15_multiple_students_marked_independently(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_15", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_15")

    faces = [
        TrackedFace(track_id=1, bbox=[10, 10, 50, 50], stable_student_id="STU001", current_status=RecognitionStatus.MATCH, current_similarity=0.85),
        TrackedFace(track_id=2, bbox=[60, 10, 100, 50], stable_student_id="STU002", current_status=RecognitionStatus.MATCH, current_similarity=0.82),
        TrackedFace(track_id=3, bbox=[110, 10, 150, 50], stable_student_id="STU003", current_status=RecognitionStatus.MATCH, current_similarity=0.89),
    ]
    records = engine.process_tracked_faces("SESS_15", faces)
    assert len(records) == 3
    db_records = db.get_attendance_for_session("SESS_15")
    assert len(db_records) == 3
    assert {r["student_id"] for r in db_records} == {"STU001", "STU002", "STU003"}


# ---------------------------------------------------------------------------
# Test 16: Temporary Face Disappearance (Attendance Persists)
# ---------------------------------------------------------------------------
def test_16_temporary_face_disappearance(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_16", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_16")

    # Mark at frame 1
    tf = TrackedFace(track_id=7, bbox=[10, 10, 50, 50], stable_student_id="STU001", current_status=RecognitionStatus.MATCH, current_similarity=0.85)
    engine.process_tracked_faces("SESS_16", [tf])
    assert len(db.get_attendance_for_session("SESS_16")) == 1

    # Empty frame (face temporarily disappeared)
    engine.process_tracked_faces("SESS_16", [])
    # Record must still be present!
    assert len(db.get_attendance_for_session("SESS_16")) == 1


# ---------------------------------------------------------------------------
# Test 17: Track ID Changes (Attendance Persists)
# ---------------------------------------------------------------------------
def test_17_track_id_changes_attendance_persists(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_17", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_17")

    # Initial track_id = 5
    tf1 = TrackedFace(track_id=5, bbox=[10, 10, 50, 50], stable_student_id="STU001", current_status=RecognitionStatus.MATCH, current_similarity=0.85)
    engine.process_tracked_faces("SESS_17", [tf1])

    # Later reappears with track_id = 19
    tf2 = TrackedFace(track_id=19, bbox=[12, 12, 52, 52], stable_student_id="STU001", current_status=RecognitionStatus.MATCH, current_similarity=0.87)
    engine.clear_cache()
    engine.process_tracked_faces("SESS_17", [tf2])

    records = db.get_attendance_for_session("SESS_17")
    assert len(records) == 1
    assert records[0]["student_id"] == "STU001"
    assert records[0]["first_track_id"] == 5
    assert records[0]["last_track_id"] == 19


# ---------------------------------------------------------------------------
# Test 18: Same Student with New Track ID Updates Record
# ---------------------------------------------------------------------------
def test_18_same_student_with_new_track_id_updates_record(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_18", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_18")

    dt1 = datetime.datetime(2026, 9, 23, 9, 1, 0)
    event1 = AttendanceEvent(session_id="SESS_18", student_id="STU004", track_id=2, timestamp=dt1.timestamp(), recognition_status=RecognitionStatus.MATCH, similarity=0.82)
    engine.process_attendance_event(event1)

    engine.clear_cache()
    dt2 = datetime.datetime(2026, 9, 23, 9, 20, 0)
    event2 = AttendanceEvent(session_id="SESS_18", student_id="STU004", track_id=22, timestamp=dt2.timestamp(), recognition_status=RecognitionStatus.MATCH, similarity=0.88)
    engine.process_attendance_event(event2)

    rec = db.get_attendance_record("SESS_18", "STU004")
    assert rec["first_track_id"] == 2
    assert rec["last_track_id"] == 22


# ---------------------------------------------------------------------------
# Test 19: Late Student Marked LATE
# ---------------------------------------------------------------------------
def test_19_late_student_marked_late(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_19", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_19", actual_start_time="2026-09-23T09:00:00")

    # Late threshold is 10 minutes (09:10:00). Student arrives at 09:15:00 (15m late)
    dt_late = datetime.datetime(2026, 9, 23, 9, 15, 0)
    event = AttendanceEvent(
        session_id="SESS_19",
        student_id="STU005",
        track_id=30,
        timestamp=dt_late.timestamp(),
        recognition_status=RecognitionStatus.MATCH,
        similarity=0.86
    )
    rec = engine.process_attendance_event(event)

    assert rec is not None
    assert rec.status == AttendanceStatus.LATE


# ---------------------------------------------------------------------------
# Test 20: Early Student Pre-Session Handling
# ---------------------------------------------------------------------------
def test_20_early_student_pre_session_handling(test_env):
    db, sm, engine, _ = test_env
    # Session starts at 09:15. Student arrives at 09:05 (within 15m pre-session window)
    sm.create_session("SESS_20", "2026-09-23", "AIDS-B", "AI", "09:15:00", "10:00:00")

    dt_early = datetime.datetime(2026, 9, 23, 9, 5, 0)
    event = AttendanceEvent(
        session_id="SESS_20",
        student_id="STU001",
        track_id=1,
        timestamp=dt_early.timestamp(),
        recognition_status=RecognitionStatus.MATCH,
        similarity=0.88
    )
    rec = engine.process_attendance_event(event)
    assert rec is not None
    assert rec.status == AttendanceStatus.PRESENT


# ---------------------------------------------------------------------------
# Test 21: Pre-Session Window Rejection if Too Early
# ---------------------------------------------------------------------------
def test_21_pre_session_window_rejection_if_outside(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_21", "2026-09-23", "AIDS-B", "AI", "09:15:00", "10:00:00")

    # Arrives at 08:30 (45 minutes early; outside 15m pre-session window)
    dt_too_early = datetime.datetime(2026, 9, 23, 8, 30, 0)
    event = AttendanceEvent(
        session_id="SESS_21",
        student_id="STU001",
        track_id=1,
        timestamp=dt_too_early.timestamp(),
        recognition_status=RecognitionStatus.MATCH,
        similarity=0.88
    )
    rec = engine.process_attendance_event(event)
    assert rec is None


# ---------------------------------------------------------------------------
# Test 22: Post-Session Window Handling
# ---------------------------------------------------------------------------
def test_22_post_session_window_handling(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_22", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_22")
    sm.end_session("SESS_22")

    # Event after session has ENDED must be rejected
    dt_post = datetime.datetime(2026, 9, 23, 9, 50, 0)
    event = AttendanceEvent(
        session_id="SESS_22",
        student_id="STU001",
        track_id=1,
        timestamp=dt_post.timestamp(),
        recognition_status=RecognitionStatus.MATCH,
        similarity=0.85
    )
    rec = engine.process_attendance_event(event)
    assert rec is None


# ---------------------------------------------------------------------------
# Test 23: Session Recovery After Interruption
# ---------------------------------------------------------------------------
def test_23_session_recovery_after_interruption(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_23", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_23")

    # Simulate crash at 09:20 and recover at 09:25 (before 09:45 planned end)
    ref_time = datetime.datetime(2026, 9, 23, 9, 25, 0)
    recovered = sm.recover_sessions(reference_time=ref_time)

    assert len(recovered) == 1
    assert recovered[0].session_id == "SESS_23"
    assert recovered[0].status == SessionState.RECOVERING

    # Can resume back to ACTIVE
    resumed = sm.resume_session("SESS_23")
    assert resumed.status == SessionState.ACTIVE


# ---------------------------------------------------------------------------
# Test 24: Application Restart Recovery (Expired Session Auto-Closed)
# ---------------------------------------------------------------------------
def test_24_application_restart_recovery(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_24", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_24")

    # Simulate app restarting at 10:15 (after 09:45 planned end)
    ref_time = datetime.datetime(2026, 9, 23, 10, 15, 0)
    recovered = sm.recover_sessions(reference_time=ref_time)

    assert len(recovered) == 1
    assert recovered[0].session_id == "SESS_24"
    assert recovered[0].status == SessionState.ENDED


# ---------------------------------------------------------------------------
# Test 25: Database Transaction Failure Rollback
# ---------------------------------------------------------------------------
def test_25_database_transaction_failure_rollback(test_env):
    db, sm, engine, _ = test_env
    # Verify that a failed multi-statement query rolls back cleanly
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO students (student_id, student_name, department, section) VALUES ('TEMP', 'T', 'CS', 'A');")
            # Intentionally cause constraint violation to trigger rollback
            cursor.execute("INSERT INTO students (student_id, student_name, department, section) VALUES ('TEMP', 'T', 'CS', 'A');")
    except sqlite3.IntegrityError:
        pass

    # 'TEMP' must not exist in database
    assert db.get_student("TEMP") is None


# ---------------------------------------------------------------------------
# Test 26: Foreign-Key Integrity (Student)
# ---------------------------------------------------------------------------
def test_26_foreign_key_integrity_student(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_26", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")

    # Non-existent student ID rejected
    event = AttendanceEvent(
        session_id="SESS_26",
        student_id="NON_EXISTENT_STU",
        track_id=1,
        timestamp=time.time(),
        recognition_status=RecognitionStatus.MATCH,
        similarity=0.9
    )
    rec = engine.process_attendance_event(event)
    assert rec is None


# ---------------------------------------------------------------------------
# Test 27: Foreign-Key Integrity (Session)
# ---------------------------------------------------------------------------
def test_27_foreign_key_integrity_session(test_env):
    db, sm, engine, _ = test_env
    # Non-existent session ID rejected
    event = AttendanceEvent(
        session_id="NON_EXISTENT_SESS",
        student_id="STU001",
        track_id=1,
        timestamp=time.time(),
        recognition_status=RecognitionStatus.MATCH,
        similarity=0.9
    )
    rec = engine.process_attendance_event(event)
    assert rec is None


# ---------------------------------------------------------------------------
# Test 28: Duplicate Session ID Prevention
# ---------------------------------------------------------------------------
def test_28_duplicate_session_id_prevention(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_28", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")

    with pytest.raises(DuplicateSessionError):
        sm.create_session("SESS_28", "2026-09-23", "AIDS-B", "AI", "10:00:00", "10:45:00")


# ---------------------------------------------------------------------------
# Test 29: Duplicate Attendance Unique Constraint
# ---------------------------------------------------------------------------
def test_29_duplicate_attendance_unique_constraint(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_29", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_29")

    # Record attendance twice
    s1, new1 = db.record_or_update_attendance("SESS_29", "STU001", "PRESENT", "2026-09-23T09:05:00", "2026-09-23T09:05:00")
    s2, new2 = db.record_or_update_attendance("SESS_29", "STU001", "PRESENT", "2026-09-23T09:05:00", "2026-09-23T09:15:00")

    assert s1 is True and new1 is True
    assert s2 is True and new2 is False # Updated, not inserted as duplicate

    rows = db.get_attendance_for_session("SESS_29")
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# Test 30: Session Date Formatting
# ---------------------------------------------------------------------------
def test_30_session_date_formatting(test_env):
    db, sm, engine, _ = test_env
    # Invalid date string rejected
    with pytest.raises(SessionManagerError):
        sm.create_session("SESS_30", "23-09-2026", "AIDS-B", "AI", "09:00:00", "09:45:00")


# ---------------------------------------------------------------------------
# Test 31: Timestamp Handling ISO Format
# ---------------------------------------------------------------------------
def test_31_timestamp_handling_iso_format(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_31", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_31")

    now = datetime.datetime.now()
    event = AttendanceEvent(
        session_id="SESS_31",
        student_id="STU001",
        track_id=1,
        timestamp=now.timestamp(),
        recognition_status=RecognitionStatus.MATCH,
        similarity=0.88
    )
    rec = engine.process_attendance_event(event)
    # Must be valid ISO string
    parsed_dt = datetime.datetime.fromisoformat(rec.first_seen)
    assert parsed_dt.year == now.year


# ---------------------------------------------------------------------------
# Test 32: Timezone Handling (Asia/Kolkata)
# ---------------------------------------------------------------------------
def test_32_timezone_handling_ist(test_env):
    db, sm, engine, _ = test_env
    assert sm.timezone_str == "Asia/Kolkata"


# ---------------------------------------------------------------------------
# Test 33: Concurrent Attendance Events (Multi-Student Frame)
# ---------------------------------------------------------------------------
def test_33_concurrent_attendance_events(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_33", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_33")

    faces = [
        TrackedFace(track_id=i, bbox=[i*10, 10, i*10+20, 30], stable_student_id=f"STU00{i}", current_status=RecognitionStatus.MATCH, current_similarity=0.85)
        for i in range(1, 6)
    ]
    records = engine.process_tracked_faces("SESS_33", faces)
    assert len(records) == 5
    assert len(db.get_attendance_for_session("SESS_33")) == 5


# ---------------------------------------------------------------------------
# Test 34: Multiple Tracks for Different Students
# ---------------------------------------------------------------------------
def test_34_multiple_tracks_for_different_students(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_34", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_34")

    t1 = TrackedFace(track_id=101, bbox=[0, 0, 50, 50], stable_student_id="STU001", current_status=RecognitionStatus.MATCH, current_similarity=0.85)
    t2 = TrackedFace(track_id=102, bbox=[60, 0, 110, 50], stable_student_id="STU002", current_status=RecognitionStatus.MATCH, current_similarity=0.88)

    engine.process_tracked_faces("SESS_34", [t1, t2])
    rec1 = db.get_attendance_record("SESS_34", "STU001")
    rec2 = db.get_attendance_record("SESS_34", "STU002")

    assert rec1["first_track_id"] == 101
    assert rec2["first_track_id"] == 102


# ---------------------------------------------------------------------------
# Test 35: Unknown Plus Known Students Together
# ---------------------------------------------------------------------------
def test_35_unknown_plus_known_students_together(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_35", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_35")

    faces = [
        TrackedFace(track_id=1, bbox=[0, 0, 50, 50], stable_student_id="STU001", current_status=RecognitionStatus.MATCH, current_similarity=0.88),
        TrackedFace(track_id=2, bbox=[60, 0, 110, 50], stable_student_id=None, current_status=RecognitionStatus.UNKNOWN, current_similarity=0.45)
    ]
    records = engine.process_tracked_faces("SESS_35", faces)
    assert len(records) == 1
    assert records[0].student_id == "STU001"


# ---------------------------------------------------------------------------
# Test 36: Recognition Instability Before Stable Result
# ---------------------------------------------------------------------------
def test_36_recognition_instability_before_stable_result(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_36", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_36")

    # Unstable / unconfirmed track has stable_student_id = None
    unstable_face = TrackedFace(
        track_id=9,
        bbox=[0, 0, 50, 50],
        stable_student_id=None,
        current_status=RecognitionStatus.MATCH, # raw match but not yet stabilized
        current_similarity=0.72
    )
    records = engine.process_tracked_faces("SESS_36", [unstable_face])
    assert len(records) == 0


# ---------------------------------------------------------------------------
# Test 37: Session Cancellation Behavior
# ---------------------------------------------------------------------------
def test_37_session_cancellation_behavior(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_37", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_37")
    sm.cancel_session("SESS_37")

    # Cannot process attendance into cancelled session
    event = AttendanceEvent(
        session_id="SESS_37", student_id="STU001", track_id=1,
        timestamp=time.time(), recognition_status=RecognitionStatus.MATCH, similarity=0.85
    )
    assert engine.process_attendance_event(event) is None


# ---------------------------------------------------------------------------
# Test 38: Ended Session Cannot Receive New Attendance
# ---------------------------------------------------------------------------
def test_38_ended_session_cannot_receive_new_attendance(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_38", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_38")
    sm.end_session("SESS_38")

    event = AttendanceEvent(
        session_id="SESS_38", student_id="STU001", track_id=1,
        timestamp=time.time(), recognition_status=RecognitionStatus.MATCH, similarity=0.85
    )
    assert engine.process_attendance_event(event) is None


# ---------------------------------------------------------------------------
# Test 39: PRESENT Status Cannot Be Demoted to LATE
# ---------------------------------------------------------------------------
def test_39_present_status_cannot_be_demoted_to_late(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_39", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_39", actual_start_time="2026-09-23T09:00:00")

    # Arrives at 09:02 -> PRESENT
    dt1 = datetime.datetime(2026, 9, 23, 9, 2, 0)
    e1 = AttendanceEvent(session_id="SESS_39", student_id="STU001", track_id=1, timestamp=dt1.timestamp(), recognition_status=RecognitionStatus.MATCH, similarity=0.85)
    engine.process_attendance_event(e1)
    rec1 = db.get_attendance_record("SESS_39", "STU001")
    assert rec1["status"] == AttendanceStatus.PRESENT

    # Re-observed at 09:35 (late time) -> Status MUST REMAIN PRESENT
    engine.clear_cache()
    dt2 = datetime.datetime(2026, 9, 23, 9, 35, 0)
    e2 = AttendanceEvent(session_id="SESS_39", student_id="STU001", track_id=1, timestamp=dt2.timestamp(), recognition_status=RecognitionStatus.MATCH, similarity=0.90)
    engine.process_attendance_event(e2)
    rec2 = db.get_attendance_record("SESS_39", "STU001")
    assert rec2["status"] == AttendanceStatus.PRESENT


# ---------------------------------------------------------------------------
# Test 40: Database Persistence After Restart
# ---------------------------------------------------------------------------
def test_40_database_persistence_after_restart(test_env):
    db, sm, engine, tmpdir = test_env
    sm.create_session("SESS_40", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_40")

    event = AttendanceEvent(session_id="SESS_40", student_id="STU001", track_id=1, timestamp=time.time(), recognition_status=RecognitionStatus.MATCH, similarity=0.85)
    engine.process_attendance_event(event)

    # Reopen database connection simulating application restart
    db_path = os.path.join(tmpdir, "test_smartclass_p8.sqlite")
    new_db = DatabaseManager(db_path=db_path)
    records = new_db.get_attendance_for_session("SESS_40")

    assert len(records) == 1
    assert records[0]["student_id"] == "STU001"


# ---------------------------------------------------------------------------
# Test 41: Session Attendance Report Reconciliation (NOT_SEEN)
# ---------------------------------------------------------------------------
def test_41_session_attendance_report_reconciliation(test_env):
    db, sm, engine, _ = test_env
    # 5 enrolled students in AIDS-B: STU001, STU002, STU003, STU004, STU005
    sm.create_session("SESS_41", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_41", actual_start_time="2026-09-23T09:00:00")

    # STU001 arrives on time (PRESENT)
    dt1 = datetime.datetime(2026, 9, 23, 9, 5, 0)
    engine.process_attendance_event(AttendanceEvent(session_id="SESS_41", student_id="STU001", track_id=1, timestamp=dt1.timestamp(), recognition_status=RecognitionStatus.MATCH, similarity=0.85))

    # STU002 arrives late (LATE)
    engine.clear_cache()
    dt2 = datetime.datetime(2026, 9, 23, 9, 25, 0)
    engine.process_attendance_event(AttendanceEvent(session_id="SESS_41", student_id="STU002", track_id=2, timestamp=dt2.timestamp(), recognition_status=RecognitionStatus.MATCH, similarity=0.85))

    # Generate Report
    report = engine.generate_session_report("SESS_41")

    assert report.total_enrolled == 5
    assert report.present_count == 1
    assert report.late_count == 1
    assert report.not_seen_count == 3
    assert report.not_seen_students == ["STU003", "STU004", "STU005"]


# ---------------------------------------------------------------------------
# Test 42: Event Deduplication Buffer Efficiency
# ---------------------------------------------------------------------------
def test_42_event_deduplication_buffer_efficiency(test_env):
    db, sm, engine, _ = test_env
    sm.create_session("SESS_42", "2026-09-23", "AIDS-B", "AI", "09:00:00", "09:45:00")
    sm.start_session("SESS_42")

    t_now = time.time()
    # Rapid stream: 30 events within 0.5s (simulating 30 FPS video frames)
    for i in range(30):
        event = AttendanceEvent(
            session_id="SESS_42",
            student_id="STU001",
            track_id=1,
            timestamp=t_now + i * 0.016, # ~16ms apart
            recognition_status=RecognitionStatus.MATCH,
            similarity=0.88
        )
        engine.process_attendance_event(event)

    # Exactly ONE attendance record exists
    records = db.get_attendance_for_session("SESS_42")
    assert len(records) == 1
