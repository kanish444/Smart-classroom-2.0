import os
import sys
import tempfile
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import create_app
from app.state import AppState, get_app_state
from app.auth import OPERATOR_TOKEN
from core.schemas import TrackedFace, RecognitionStatus, AttendanceStatus, SessionState


@pytest.fixture
def client_env():
    """Provides an isolated AppState, temporary database, and FastAPI TestClient."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test_p9.sqlite")
        state = AppState(db_path=test_db)

        # Populate standard synthetic students
        state.db.add_student("STU001", "Alice Smith", "Computer Science", "AIDS-B")
        state.db.add_student("STU002", "Bob Jones", "Computer Science", "AIDS-B")
        state.db.add_student("STU003", "Charlie Brown", "Computer Science", "AIDS-B")
        state.db.add_student("STU004", "Diana Prince", "Computer Science", "AIDS-B")
        state.db.add_student("STU005", "Evan Wright", "Computer Science", "AIDS-B")
        state.db.add_student("STU006", "Fiona Gallagher", "Computer Science", "AIDS-A")

        app = create_app()
        app.dependency_overrides[get_app_state] = lambda: state
        client = TestClient(app)

        yield client, state, tmpdir


# ---------------------------------------------------------------------------
# Test 1: Dashboard HTML Root Loading
# ---------------------------------------------------------------------------
def test_01_dashboard_root_html(client_env):
    client, _, _ = client_env
    res = client.get("/")
    assert res.status_code == 200
    assert "SmartClass Vision AI" in res.text
    assert "Smart Board" in res.text


# ---------------------------------------------------------------------------
# Test 2: Static CSS Assets Served
# ---------------------------------------------------------------------------
def test_02_static_css_served(client_env):
    client, _, _ = client_env
    res = client.get("/static/css/dashboard.css")
    assert res.status_code == 200
    assert "SmartClass Vision AI" in res.text
    assert "--bg-primary" in res.text


# ---------------------------------------------------------------------------
# Test 3: Static JS Assets Served
# ---------------------------------------------------------------------------
def test_03_static_js_served(client_env):
    client, _, _ = client_env
    res = client.get("/static/js/dashboard.js")
    assert res.status_code == 200
    assert "SmartClass Vision AI" in res.text
    assert "initSSE" in res.text


# ---------------------------------------------------------------------------
# Test 4: Health Endpoint Structure
# ---------------------------------------------------------------------------
def test_04_health_endpoint_structure(client_env):
    client, _, _ = client_env
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert "timestamp" in body
    data = body["data"]
    assert data["status"] in ("ONLINE", "DEGRADED", "OFFLINE")
    for key in ("camera", "detection", "recognition", "tracking", "attendance", "database", "api"):
        assert key in data["components"]
        assert data["components"][key]["status"] in ("ONLINE", "DEGRADED", "OFFLINE")


# ---------------------------------------------------------------------------
# Test 5: Health Database Latency Metric
# ---------------------------------------------------------------------------
def test_05_health_database_latency(client_env):
    client, _, _ = client_env
    res = client.get("/api/health")
    data = res.json()["data"]
    db_comp = data["components"]["database"]
    assert db_comp["status"] == "ONLINE"
    assert db_comp["latency_ms"] is not None
    assert db_comp["latency_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Test 6: Current Session Null When None Active
# ---------------------------------------------------------------------------
def test_06_current_session_null(client_env):
    client, _, _ = client_env
    res = client.get("/api/session/current")
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["data"] is None


# ---------------------------------------------------------------------------
# Test 7: Current Session Active
# ---------------------------------------------------------------------------
def test_07_current_session_active(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_P9_01", "2026-09-23", "AIDS-B", "Python", "09:00", "10:00")
    state.session_manager.start_session("SESS_P9_01")

    res = client.get("/api/session/current")
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    sess = body["data"]
    assert sess["session_id"] == "SESS_P9_01"
    assert sess["status"] == "ACTIVE"
    assert sess["class_section"] == "AIDS-B"


# ---------------------------------------------------------------------------
# Test 8: List Sessions
# ---------------------------------------------------------------------------
def test_08_list_sessions_all(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_A", "2026-09-23", "AIDS-B", "Sub1", "09:00", "10:00")
    state.session_manager.create_session("SESS_B", "2026-09-23", "AIDS-B", "Sub2", "10:15", "11:15")

    res = client.get("/api/sessions")
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    items = body["data"]["items"]
    assert len(items) == 2
    assert body["data"]["pagination"]["total_items"] == 2


# ---------------------------------------------------------------------------
# Test 9: List Sessions Pagination
# ---------------------------------------------------------------------------
def test_09_list_sessions_pagination(client_env):
    client, state, _ = client_env
    for i in range(5):
        state.session_manager.create_session(f"SESS_{i}", "2026-09-23", "AIDS-B", "Sub", "09:00", "10:00")

    res = client.get("/api/sessions?page=1&page_size=2")
    assert res.status_code == 200
    body = res.json()["data"]
    assert len(body["items"]) == 2
    assert body["pagination"]["page"] == 1
    assert body["pagination"]["page_size"] == 2
    assert body["pagination"]["total_items"] == 5
    assert body["pagination"]["total_pages"] == 3


# ---------------------------------------------------------------------------
# Test 10: List Sessions Date Filter
# ---------------------------------------------------------------------------
def test_10_list_sessions_date_filter(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_D1", "2026-09-23", "AIDS-B", "Sub1", "09:00", "10:00")
    state.session_manager.create_session("SESS_D2", "2026-09-24", "AIDS-B", "Sub2", "09:00", "10:00")

    res = client.get("/api/sessions?date=2026-09-23")
    assert res.status_code == 200
    items = res.json()["data"]["items"]
    assert len(items) == 1
    assert items[0]["session_id"] == "SESS_D1"


# ---------------------------------------------------------------------------
# Test 11: List Sessions Status Filter
# ---------------------------------------------------------------------------
def test_11_list_sessions_status_filter(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_STAT1", "2026-09-23", "AIDS-B", "Sub1", "09:00", "10:00")
    state.session_manager.create_session("SESS_STAT2", "2026-09-23", "AIDS-B", "Sub2", "10:00", "11:00")
    state.session_manager.start_session("SESS_STAT1")

    res = client.get("/api/sessions?status=ACTIVE")
    assert res.status_code == 200
    items = res.json()["data"]["items"]
    assert len(items) == 1
    assert items[0]["session_id"] == "SESS_STAT1"


# ---------------------------------------------------------------------------
# Test 12: Session Detail by ID
# ---------------------------------------------------------------------------
def test_12_session_by_id_success(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_SPECIFIC", "2026-09-23", "AIDS-B", "ML", "09:00", "10:00")

    res = client.get("/api/sessions/SESS_SPECIFIC")
    assert res.status_code == 200
    sess = res.json()["data"]
    assert sess["session_id"] == "SESS_SPECIFIC"
    assert sess["subject"] == "ML"


# ---------------------------------------------------------------------------
# Test 13: Session Detail Not Found
# ---------------------------------------------------------------------------
def test_13_session_by_id_not_found(client_env):
    client, _, _ = client_env
    res = client.get("/api/sessions/NONEXISTENT_SESSION")
    assert res.status_code == 404
    body = res.json()
    assert body["success"] is False
    assert body["error"]["code"] == "HTTP_404"
    assert "not found" in body["error"]["message"].lower()


# ---------------------------------------------------------------------------
# Test 14: Create Session Authorized
# ---------------------------------------------------------------------------
def test_14_create_session_authorized(client_env):
    client, _, _ = client_env
    payload = {
        "session_id": "SESS_NEW_API",
        "date": "2026-09-23",
        "class_section": "AIDS-B",
        "subject": "Deep Learning",
        "planned_start_time": "11:00",
        "planned_end_time": "12:00"
    }
    res = client.post("/api/sessions", json=payload, headers={"X-Operator-Token": OPERATOR_TOKEN})
    assert res.status_code == 201
    body = res.json()
    assert body["success"] is True
    assert body["data"]["session_id"] == "SESS_NEW_API"
    assert body["data"]["status"] == "SCHEDULED"


# ---------------------------------------------------------------------------
# Test 15: Create Session Unauthorized (Operator Protection)
# ---------------------------------------------------------------------------
def test_15_create_session_unauthorized(client_env):
    client, _, _ = client_env
    payload = {
        "session_id": "SESS_UNAUTH",
        "date": "2026-09-23",
        "class_section": "AIDS-B",
        "subject": "Deep Learning",
        "planned_start_time": "11:00",
        "planned_end_time": "12:00"
    }
    res = client.post("/api/sessions", json=payload)
    assert res.status_code == 403
    body = res.json()
    assert body["success"] is False
    assert "Operator authorization required" in body["error"]["message"]


# ---------------------------------------------------------------------------
# Test 16: Create Session Duplicate Rejected
# ---------------------------------------------------------------------------
def test_16_create_session_duplicate_rejected(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_DUP", "2026-09-23", "AIDS-B", "Math", "09:00", "10:00")

    payload = {
        "session_id": "SESS_DUP",
        "date": "2026-09-23",
        "class_section": "AIDS-B",
        "subject": "Math",
        "planned_start_time": "09:00",
        "planned_end_time": "10:00"
    }
    res = client.post("/api/sessions", json=payload, headers={"X-Operator-Token": OPERATOR_TOKEN})
    assert res.status_code == 409
    body = res.json()
    assert body["success"] is False


# ---------------------------------------------------------------------------
# Test 17: Start Session Lifecycle
# ---------------------------------------------------------------------------
def test_17_start_session_lifecycle(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_START_TEST", "2026-09-23", "AIDS-B", "Math", "09:00", "10:00")

    res = client.post("/api/sessions/SESS_START_TEST/start", headers={"X-Operator-Token": OPERATOR_TOKEN})
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["data"]["status"] == "ACTIVE"


# ---------------------------------------------------------------------------
# Test 18: Pause and Resume Session
# ---------------------------------------------------------------------------
def test_18_pause_and_resume_session(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_PR", "2026-09-23", "AIDS-B", "Math", "09:00", "10:00")
    state.session_manager.start_session("SESS_PR")

    # Pause
    res = client.post("/api/sessions/SESS_PR/pause", headers={"X-Operator-Token": OPERATOR_TOKEN})
    assert res.status_code == 200
    assert res.json()["data"]["status"] == "PAUSED"

    # Resume
    res2 = client.post("/api/sessions/SESS_PR/resume", headers={"X-Operator-Token": OPERATOR_TOKEN})
    assert res2.status_code == 200
    assert res2.json()["data"]["status"] == "ACTIVE"


# ---------------------------------------------------------------------------
# Test 19: End Session Lifecycle
# ---------------------------------------------------------------------------
def test_19_end_session_lifecycle(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_END_TEST", "2026-09-23", "AIDS-B", "Math", "09:00", "10:00")
    state.session_manager.start_session("SESS_END_TEST")

    res = client.post(
        "/api/sessions/SESS_END_TEST/end",
        json={"confirm": True},
        headers={"X-Operator-Token": OPERATOR_TOKEN}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["data"]["status"] == "ENDED"


# ---------------------------------------------------------------------------
# Test 20: End Session Without Confirmation Rejected
# ---------------------------------------------------------------------------
def test_20_end_session_without_confirmation_rejected(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_NOCONFIRM", "2026-09-23", "AIDS-B", "Math", "09:00", "10:00")
    state.session_manager.start_session("SESS_NOCONFIRM")

    res = client.post(
        "/api/sessions/SESS_NOCONFIRM/end",
        json={"confirm": False},
        headers={"X-Operator-Token": OPERATOR_TOKEN}
    )
    assert res.status_code == 400
    assert "requires explicit confirmation" in res.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Test 21: Invalid State Transition Rejected
# ---------------------------------------------------------------------------
def test_21_invalid_transition_rejected(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_TERM", "2026-09-23", "AIDS-B", "Math", "09:00", "10:00")
    state.session_manager.start_session("SESS_TERM")
    state.session_manager.end_session("SESS_TERM")

    # Attempt to pause an ended session
    res = client.post("/api/sessions/SESS_TERM/pause", headers={"X-Operator-Token": OPERATOR_TOKEN})
    assert res.status_code == 400
    assert "Invalid session state transition" in res.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Test 22: Attendance Current Session
# ---------------------------------------------------------------------------
def test_22_attendance_current_session(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_ATT_CUR", "2026-09-23", "AIDS-B", "Math", "09:00", "10:00")
    state.session_manager.start_session("SESS_ATT_CUR")

    # Record attendance for STU001
    state.db.record_or_update_attendance(
        session_id="SESS_ATT_CUR",
        student_id="STU001",
        status="PRESENT",
        first_seen="2026-09-23T09:05:00",
        last_seen="2026-09-23T09:10:00",
        similarity=0.85
    )

    res = client.get("/api/attendance/current")
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["session"]["session_id"] == "SESS_ATT_CUR"
    assert data["present_count"] == 1
    assert len(data["records"]) == 1
    assert data["records"][0]["student_id"] == "STU001"


# ---------------------------------------------------------------------------
# Test 23: Attendance by Session ID
# ---------------------------------------------------------------------------
def test_23_attendance_by_session_id(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_HIST", "2026-09-23", "AIDS-B", "Math", "09:00", "10:00")
    state.db.record_or_update_attendance(
        session_id="SESS_HIST",
        student_id="STU002",
        status="LATE",
        first_seen="2026-09-23T09:25:00",
        last_seen="2026-09-23T09:25:00",
        similarity=0.78
    )

    res = client.get("/api/attendance/session/SESS_HIST")
    assert res.status_code == 200
    report = res.json()["data"]
    assert report["late_count"] == 1
    assert report["records"][0]["status"] == "LATE"


# ---------------------------------------------------------------------------
# Test 24: Attendance Session Not Found
# ---------------------------------------------------------------------------
def test_24_attendance_session_not_found(client_env):
    client, _, _ = client_env
    res = client.get("/api/attendance/session/MISSING_SESS")
    assert res.status_code == 404
    assert res.json()["success"] is False


# ---------------------------------------------------------------------------
# Test 25: Students List Pagination
# ---------------------------------------------------------------------------
def test_25_students_list_pagination(client_env):
    client, _, _ = client_env
    res = client.get("/api/students?page=1&page_size=3")
    assert res.status_code == 200
    body = res.json()["data"]
    assert len(body["items"]) == 3
    assert body["pagination"]["page"] == 1
    assert body["pagination"]["total_items"] == 6


# ---------------------------------------------------------------------------
# Test 26: Students Search by Name
# ---------------------------------------------------------------------------
def test_26_students_search_by_name(client_env):
    client, _, _ = client_env
    res = client.get("/api/students?query=Alice")
    assert res.status_code == 200
    items = res.json()["data"]["items"]
    assert len(items) == 1
    assert items[0]["student_name"] == "Alice Smith"


# ---------------------------------------------------------------------------
# Test 27: Students Search by ID
# ---------------------------------------------------------------------------
def test_27_students_search_by_id(client_env):
    client, _, _ = client_env
    res = client.get("/api/students?query=STU003")
    assert res.status_code == 200
    items = res.json()["data"]["items"]
    assert len(items) == 1
    assert items[0]["student_id"] == "STU003"


# ---------------------------------------------------------------------------
# Test 28: Students Filter by Section
# ---------------------------------------------------------------------------
def test_28_students_filter_by_section(client_env):
    client, _, _ = client_env
    res = client.get("/api/students?section=AIDS-A")
    assert res.status_code == 200
    items = res.json()["data"]["items"]
    assert len(items) == 1
    assert items[0]["student_id"] == "STU006"


# ---------------------------------------------------------------------------
# Test 29: Student Detail Found
# ---------------------------------------------------------------------------
def test_29_student_detail_found(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_REC", "2026-09-23", "AIDS-B", "Math", "09:00", "10:00")
    state.db.record_or_update_attendance(
        session_id="SESS_REC",
        student_id="STU001",
        status="PRESENT",
        first_seen="2026-09-23T09:02:00",
        last_seen="2026-09-23T09:02:00",
        similarity=0.88
    )

    res = client.get("/api/students/STU001")
    assert res.status_code == 200
    detail = res.json()["data"]
    assert detail["profile"]["student_id"] == "STU001"
    assert detail["total_sessions_attended"] == 1
    assert len(detail["attendance_history"]) == 1


# ---------------------------------------------------------------------------
# Test 30: Student Detail Not Found
# ---------------------------------------------------------------------------
def test_30_student_detail_not_found(client_env):
    client, _, _ = client_env
    res = client.get("/api/students/STU999_NONEXISTENT")
    assert res.status_code == 404
    assert res.json()["success"] is False


# ---------------------------------------------------------------------------
# Test 31: Dashboard Summary With Active Session
# ---------------------------------------------------------------------------
def test_31_dashboard_summary_with_active_session(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_DASH", "2026-09-23", "AIDS-B", "Vision", "09:00", "10:00")
    state.session_manager.start_session("SESS_DASH")
    state.db.record_or_update_attendance(
        session_id="SESS_DASH",
        student_id="STU001",
        status="PRESENT",
        first_seen="2026-09-23T09:03:00",
        last_seen="2026-09-23T09:03:00",
        similarity=0.82
    )

    res = client.get("/api/dashboard/summary")
    assert res.status_code == 200
    sum_data = res.json()["data"]
    assert sum_data["session_status"] == "ACTIVE"
    assert sum_data["session"]["session_id"] == "SESS_DASH"
    assert sum_data["present_count"] == 1
    assert sum_data["total_enrolled"] == 5  # AIDS-B has 5 students


# ---------------------------------------------------------------------------
# Test 32: Dashboard Summary No Active Session
# ---------------------------------------------------------------------------
def test_32_dashboard_summary_no_active_session(client_env):
    client, _, _ = client_env
    res = client.get("/api/dashboard/summary")
    assert res.status_code == 200
    sum_data = res.json()["data"]
    assert sum_data["session_status"] == "NO_ACTIVE_SESSION"
    assert sum_data["session"] is None
    assert sum_data["present_count"] == 0
    assert sum_data["total_enrolled"] == 6


# ---------------------------------------------------------------------------
# Test 33: Active Tracks Endpoint
# ---------------------------------------------------------------------------
def test_33_active_tracks_endpoint(client_env):
    client, state, _ = client_env
    tf = TrackedFace(
        track_id=101,
        bbox=[100, 100, 200, 200],
        state="ACTIVE",
        stable_student_id="STU001",
        stable_student_name="Alice Smith",
        current_similarity=0.89,
        current_status=RecognitionStatus.MATCH
    )
    state.update_telemetry(frame=None, tracks=[tf], fps=28.5)

    res = client.get("/api/tracks/active")
    assert res.status_code == 200
    tracks = res.json()["data"]
    assert len(tracks) == 1
    assert tracks[0]["track_id"] == 101
    assert tracks[0]["stable_student_id"] == "STU001"
    assert tracks[0]["current_similarity"] == 0.89


# ---------------------------------------------------------------------------
# Test 34: Recognition Status Endpoint
# ---------------------------------------------------------------------------
def test_34_recognition_status_endpoint(client_env):
    client, _, _ = client_env
    res = client.get("/api/recognition/status")
    assert res.status_code == 200
    data = res.json()["data"]
    assert "model_name" in data
    assert data["embedding_dim"] == 512
    assert data["similarity_threshold"] > 0.0
    assert data["status"] == "ONLINE"


# ---------------------------------------------------------------------------
# Test 35: CSV Export Format and Headers
# ---------------------------------------------------------------------------
def test_35_csv_export_format_and_headers(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_CSV", "2026-09-23", "AIDS-B", "Math", "09:00", "10:00")
    state.db.record_or_update_attendance(
        session_id="SESS_CSV",
        student_id="STU001",
        status="PRESENT",
        first_seen="2026-09-23T09:05:00",
        last_seen="2026-09-23T09:05:00",
        similarity=0.84
    )

    res = client.get("/api/export/attendance/SESS_CSV/csv")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/csv")
    assert 'attachment; filename="attendance_SESS_CSV.csv"' in res.headers["content-disposition"]

    lines = res.text.strip().split("\r\n")
    if len(lines) == 1:
        lines = res.text.strip().split("\n")
    headers = [h.strip() for h in lines[0].split(",")]
    expected_headers = [
        "Session ID", "Date", "Class", "Subject", "Student ID",
        "Register Number", "Student Name", "Status", "First Seen", "Last Seen"
    ]
    assert headers == expected_headers


# ---------------------------------------------------------------------------
# Test 36: CSV Export Content Correctness
# ---------------------------------------------------------------------------
def test_36_csv_export_content(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_CSV_CONT", "2026-09-23", "AIDS-B", "Python", "09:00", "10:00")
    state.db.record_or_update_attendance(
        session_id="SESS_CSV_CONT",
        student_id="STU002",
        status="LATE",
        first_seen="2026-09-23T09:20:00",
        last_seen="2026-09-23T09:30:00",
        similarity=0.76
    )

    res = client.get("/api/export/attendance/SESS_CSV_CONT/csv")
    assert "SESS_CSV_CONT" in res.text
    assert "STU002" in res.text
    assert "Bob Jones" in res.text
    assert "LATE" in res.text


# ---------------------------------------------------------------------------
# Test 37: CSV Export Safety - No Biometric Embeddings
# ---------------------------------------------------------------------------
def test_37_csv_export_safety_no_embeddings(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_SAFE", "2026-09-23", "AIDS-B", "Math", "09:00", "10:00")
    state.db.record_or_update_attendance(
        session_id="SESS_SAFE",
        student_id="STU001",
        status="PRESENT",
        first_seen="2026-09-23T09:01:00",
        last_seen="2026-09-23T09:01:00",
        similarity=0.91
    )

    res = client.get("/api/export/attendance/SESS_SAFE/csv")
    content = res.text.lower()
    # Ensure no vector arrays, embedding blobs, or biometric arrays are exposed
    assert "vector_blob" not in content
    assert "embedding" not in content
    assert "faiss" not in content
    assert "0x" not in content


# ---------------------------------------------------------------------------
# Test 38: JSON Export Structure
# ---------------------------------------------------------------------------
def test_38_json_export_structure(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_JSON", "2026-09-23", "AIDS-B", "Vision", "09:00", "10:00")
    state.db.record_or_update_attendance(
        session_id="SESS_JSON",
        student_id="STU001",
        status="PRESENT",
        first_seen="2026-09-23T09:02:00",
        last_seen="2026-09-23T09:02:00",
        similarity=0.88
    )

    res = client.get("/api/export/attendance/SESS_JSON/json")
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    data = body["data"]
    assert "session" in data
    assert data["session"]["session_id"] == "SESS_JSON"
    assert "attendance" in data
    assert len(data["attendance"]) == 1
    assert data["attendance"][0]["student_id"] == "STU001"
    assert data["attendance"][0]["name"] == "Alice Smith"


# ---------------------------------------------------------------------------
# Test 39: JSON Export Safety - No Biometric Data
# ---------------------------------------------------------------------------
def test_39_json_export_safety_no_embeddings(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_JSON_SAFE", "2026-09-23", "AIDS-B", "Vision", "09:00", "10:00")
    res = client.get("/api/export/attendance/SESS_JSON_SAFE/json")
    content = res.text.lower()
    assert "vector" not in content
    assert "blob" not in content
    assert "faiss" not in content


# ---------------------------------------------------------------------------
# Test 40: Export Invalid Session Handled
# ---------------------------------------------------------------------------
def test_40_export_invalid_session(client_env):
    client, _, _ = client_env
    res_csv = client.get("/api/export/attendance/NONEXISTENT/csv")
    assert res_csv.status_code == 404

    res_json = client.get("/api/export/attendance/NONEXISTENT/json")
    assert res_json.status_code == 404


# ---------------------------------------------------------------------------
# Test 41: Empty Attendance Export
# ---------------------------------------------------------------------------
def test_41_empty_attendance_export(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_EMPTY", "2026-09-23", "AIDS-B", "Math", "09:00", "10:00")

    # CSV with 0 attendance
    res_csv = client.get("/api/export/attendance/SESS_EMPTY/csv")
    assert res_csv.status_code == 200
    lines = res_csv.text.strip().split("\n")
    assert len(lines) == 1  # Only header line

    # JSON with 0 attendance
    res_json = client.get("/api/export/attendance/SESS_EMPTY/json")
    assert res_json.status_code == 200
    assert len(res_json.json()["data"]["attendance"]) == 0


# ---------------------------------------------------------------------------
# Test 42: Unknown Track Safe Handling (No Guessing)
# ---------------------------------------------------------------------------
def test_42_unknown_track_safe_handling(client_env):
    client, state, _ = client_env
    # Add an unconfirmed track
    tf_unknown = TrackedFace(
        track_id=202,
        bbox=[50, 50, 150, 150],
        state="ACTIVE",
        stable_student_id=None,
        stable_student_name=None,
        current_similarity=0.42,
        current_status=RecognitionStatus.UNKNOWN
    )
    state.update_telemetry(frame=None, tracks=[tf_unknown], unknown_count=1)

    res = client.get("/api/dashboard/summary")
    assert res.status_code == 200
    summary = res.json()["data"]
    assert summary["unknown_count"] == 1
    assert len(summary["active_tracks"]) == 1
    # Bounded track must reflect safe UNKNOWN
    assert summary["active_tracks"][0]["stable_student_id"] is None
    assert summary["active_tracks"][0]["current_status"] == RecognitionStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Test 43: Attendance Count Correctness
# ---------------------------------------------------------------------------
def test_43_attendance_count_correctness(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_COUNTS", "2026-09-23", "AIDS-B", "Math", "09:00", "10:00")
    state.session_manager.start_session("SESS_COUNTS")

    # Mark 1 Present, 1 Late
    state.db.record_or_update_attendance("SESS_COUNTS", "STU001", "PRESENT", "2026-09-23T09:05:00", "2026-09-23T09:05:00")
    state.db.record_or_update_attendance("SESS_COUNTS", "STU002", "LATE", "2026-09-23T09:20:00", "2026-09-23T09:20:00")

    res = client.get("/api/dashboard/summary")
    data = res.json()["data"]
    assert data["present_count"] == 1
    assert data["late_count"] == 1
    # 5 enrolled in AIDS-B: 5 - 2 = 3 not seen
    assert data["not_seen_count"] == 3
    assert data["present_count"] + data["late_count"] + data["not_seen_count"] == data["total_enrolled"]


# ---------------------------------------------------------------------------
# Test 44: Camera Offline Telemetry
# ---------------------------------------------------------------------------
def test_44_camera_offline_telemetry(client_env):
    client, state, _ = client_env
    with state._lock:
        state.camera_online = False

    res = client.get("/api/health")
    data = res.json()["data"]
    assert data["components"]["camera"]["status"] == "OFFLINE"
    assert data["status"] in ("DEGRADED", "OFFLINE")

    # Reset
    with state._lock:
        state.camera_online = True


# ---------------------------------------------------------------------------
# Test 45: API Error Response Envelope
# ---------------------------------------------------------------------------
def test_45_api_error_response_envelope(client_env):
    client, _, _ = client_env
    # Invalid endpoint
    res = client.get("/api/sessions/INVALID_404")
    assert res.status_code == 404
    body = res.json()
    assert body["success"] is False
    assert body["error"]["code"] == "HTTP_404"
    assert "timestamp" in body


# ---------------------------------------------------------------------------
# Test 46: Video Feed Stream Content Type
# ---------------------------------------------------------------------------
def test_46_video_feed_stream(client_env):
    client, _, _ = client_env
    # Request single frame with limit=1
    res = client.get("/api/video/feed?view_mode=normal&limit=1")
    assert res.status_code == 200
    content_type = res.headers["content-type"]
    assert "multipart/x-mixed-replace" in content_type
    assert b"--frame" in res.content


# ---------------------------------------------------------------------------
# Test 47: Operator Token via Query Parameter
# ---------------------------------------------------------------------------
def test_47_operator_token_query_param(client_env):
    client, state, _ = client_env
    state.session_manager.create_session("SESS_QUERY_AUTH", "2026-09-23", "AIDS-B", "Math", "09:00", "10:00")

    # Authorize using query param instead of header
    res = client.post(f"/api/sessions/SESS_QUERY_AUTH/start?operator_token={OPERATOR_TOKEN}")
    assert res.status_code == 200
    assert res.json()["data"]["status"] == "ACTIVE"


# ---------------------------------------------------------------------------
# Test 48: Server-Sent Events Stream Delivery
# ---------------------------------------------------------------------------
def test_48_sse_events_stream(client_env):
    client, _, _ = client_env
    res = client.get("/api/events/sse?limit=1")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    assert "data:" in res.text
    assert "total_enrolled" in res.text

