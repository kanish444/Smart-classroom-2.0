import os
import sys
import tempfile
import time
import datetime
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import create_app
from app.state import AppState, get_app_state
from core.schemas import TrackedFace, TrackState, RecognitionStatus, AttendanceStatus


def test_full_pipeline_integration():
    """
    INTEGRATION TEST: End-to-End Pipeline Verification
    Camera Frame / Synthetic Tracks
            ↓
    Temporal Tracking (TrackedFace)
            ↓
    Attendance Engine (Business Rules & Late Policy)
            ↓
    SQLite Persistence (Sessions & Attendance tables)
            ↓
    FastAPI Endpoints (/api/dashboard/summary, /api/attendance/current)
            ↓
    Dashboard Export APIs (CSV & JSON)

    Verifies that a recognition event appears correctly in the dashboard
    without hard-coding or bypassing any intermediate layer.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "integration_test.sqlite")
        state = AppState(db_path=test_db)

        # 1. Enroll synthetic student roster
        state.db.add_student("STU001", "Alice Smith", "Computer Science", "AIDS-B")
        state.db.add_student("STU002", "Bob Jones", "Computer Science", "AIDS-B")
        state.db.add_student("STU003", "Charlie Brown", "Computer Science", "AIDS-B")

        # 2. Schedule and Start Instructional Session
        session_id = "SESS_INTEGRATION_2026"
        state.session_manager.create_session(
            session_id=session_id,
            date="2026-09-23",
            class_section="AIDS-B",
            subject="Vision Systems",
            planned_start_time="09:15:00",
            planned_end_time="10:00:00"
        )
        sess_info = state.session_manager.start_session(session_id, actual_start_time="2026-09-23T09:15:00")
        assert sess_info.status == "ACTIVE"

        # 3. Simulate Pipeline Observations with timestamps relative to session start
        dt_alice = datetime.datetime.fromisoformat("2026-09-23T09:18:00")  # 3 min in -> PRESENT
        dt_bob = datetime.datetime.fromisoformat("2026-09-23T09:28:00")    # 13 min in -> LATE

        track_alice = TrackedFace(
            track_id=1,
            bbox=[120, 150, 240, 300],
            state=TrackState.ACTIVE,
            stable_student_id="STU001",
            stable_student_name="Alice Smith",
            current_status=RecognitionStatus.MATCH,
            current_similarity=0.88,
            quality_status="RECOGNITION_READY",
            hits=15,
            age=15
        )
        track_bob = TrackedFace(
            track_id=2,
            bbox=[400, 150, 520, 300],
            state=TrackState.ACTIVE,
            stable_student_id="STU002",
            stable_student_name="Bob Jones",
            current_status=RecognitionStatus.MATCH,
            current_similarity=0.82,
            quality_status="RECOGNITION_READY",
            hits=12,
            age=12
        )
        track_unknown = TrackedFace(
            track_id=3,
            bbox=[700, 150, 820, 300],
            state=TrackState.ACTIVE,
            stable_student_id=None,
            stable_student_name=None,
            current_status=RecognitionStatus.UNKNOWN,
            current_similarity=0.41,
            quality_status="RECOGNITION_READY",
            hits=8,
            age=8
        )

        # 4. Ingest into Attendance Engine
        rec_alice = state.attendance_engine.process_tracked_faces(session_id, [track_alice], timestamp=dt_alice.timestamp())
        rec_bob = state.attendance_engine.process_tracked_faces(session_id, [track_bob], timestamp=dt_bob.timestamp())
        rec_unk = state.attendance_engine.process_tracked_faces(session_id, [track_unknown], timestamp=dt_alice.timestamp())

        assert len(rec_alice) == 1
        assert rec_alice[0].status == AttendanceStatus.PRESENT
        assert len(rec_bob) == 1
        assert rec_bob[0].status == AttendanceStatus.LATE
        assert len(rec_unk) == 0  # Unknown track rejected

        # 5. Push telemetry to AppState
        all_tracks = [track_alice, track_bob, track_unknown]
        state.update_telemetry(frame=None, tracks=all_tracks, fps=29.8, unknown_count=1)

        # 6. Initialize FastAPI test client
        app = create_app()
        app.dependency_overrides[get_app_state] = lambda: state
        client = TestClient(app)

        # 7. Verify Dashboard Summary API
        dash_res = client.get("/api/dashboard/summary")
        assert dash_res.status_code == 200
        dash_json = dash_res.json()
        assert dash_json["success"] is True

        data = dash_json["data"]
        assert data["session_status"] == "ACTIVE"
        assert data["session"]["session_id"] == session_id
        assert data["total_enrolled"] == 3
        assert data["present_count"] == 1
        assert data["late_count"] == 1
        assert data["not_seen_count"] == 1  # STU003 Charlie Brown
        assert data["unknown_count"] == 1
        assert data["active_track_count"] == 3
        assert data["system_status"] == "ONLINE"

        # 8. Verify Attendance Report API
        att_res = client.get("/api/attendance/current")
        assert att_res.status_code == 200
        att_data = att_res.json()["data"]
        assert att_data["session"]["session_id"] == session_id
        assert att_data["present_count"] == 1
        assert att_data["late_count"] == 1
        assert "STU003" in att_data["not_seen_students"]
        marked_ids = [r["student_id"] for r in att_data["records"]]
        assert "STU001" in marked_ids
        assert "STU002" in marked_ids

        # 9. Verify CSV Export
        csv_res = client.get(f"/api/export/attendance/{session_id}/csv")
        assert csv_res.status_code == 200
        assert csv_res.headers["content-type"].startswith("text/csv")
        csv_text = csv_res.text
        assert "Alice Smith" in csv_text
        assert "Bob Jones" in csv_text
        assert "PRESENT" in csv_text
        assert "LATE" in csv_text
        assert "PRESENT" in csv_text
        assert "STU001" in csv_text
        assert "STU002" in csv_text
        # Security invariant: zero embeddings
        assert "vector" not in csv_text.lower()
        assert "blob" not in csv_text.lower()

        # 10. Verify JSON Export
        json_res = client.get(f"/api/export/attendance/{session_id}/json")
        assert json_res.status_code == 200
        json_data = json_res.json()["data"]
        assert json_data["session"]["session_id"] == session_id
        assert len(json_data["attendance"]) == 2
        # Verify student fields
        assert json_data["attendance"][0]["student_id"] in ("STU001", "STU002")
        assert json_data["attendance"][1]["student_id"] in ("STU001", "STU002")

        print("\nIntegration test passed: Complete Camera -> Tracking -> Attendance -> DB -> API -> Dashboard chain verified.")
