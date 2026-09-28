import os
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.user_service import UserService
from app.models.user import UserModel, UserRole, UserStatus
from app.core.security import hash_password, create_access_token


@pytest.fixture(scope="module")
def flow_client(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("flow_test")
    test_db = str(temp_dir / "smartclass.sqlite")
    
    user_svc = UserService(db_path=test_db, mongo_uri="mongodb://localhost:27017")
    
    # Create Advisor ADV001
    adv = UserModel(
        user_id="ADV001",
        name="Prof. Ramesh Kumar",
        role=UserRole.CLASS_ADVISOR,
        password_hash=hash_password("password123"),
        email="ramesh@college.edu",
        department="AI&DS",
        year="3rd Year",
        section="B",
        assigned_classroom="AIDS-B",
        status=UserStatus.ACTIVE
    )
    user_svc.create_user(adv)
    
    # Create Advisor without class
    adv_unassigned = UserModel(
        user_id="ADV002",
        name="Prof. Unassigned",
        role=UserRole.CLASS_ADVISOR,
        password_hash=hash_password("password123"),
        email="unassigned@college.edu",
        department="",
        year="",
        section="",
        assigned_classroom="",
        status=UserStatus.ACTIVE
    )
    user_svc.create_user(adv_unassigned)

    app = create_app()
    from app.auth import get_user_service
    app.dependency_overrides[get_user_service] = lambda: user_svc

    with TestClient(app) as test_client:
        yield {
            "client": test_client,
            "user_service": user_svc,
            "db_path": test_db
        }


def test_01_root_and_login_serve_login_page_no_bypass(flow_client):
    """
    MASTER FLOW 1:
    Open Web Server (/) or (/login) without authentication:
    - Must serve login page directly
    - Must NOT open normal dashboard automatically
    - Must NOT bypass login
    """
    c = flow_client["client"]
    resp = c.get("/", follow_redirects=False)
    assert resp.status_code == 200
    assert "Secure Portal Login" in resp.text
    assert "User ID / Email" in resp.text
    assert "loginForm" in resp.text

    resp_login = c.get("/login", follow_redirects=False)
    assert resp_login.status_code == 200
    assert "Secure Portal Login" in resp_login.text

    # Unauthenticated /dashboard must redirect to /login (no bypass)
    resp_dash = c.get("/dashboard", follow_redirects=False)
    assert resp_dash.status_code in (302, 303, 307)
    assert resp_dash.headers["location"] == "/login"


def test_02_authenticate_hod_by_user_id_and_email(flow_client):
    """
    MASTER FLOW 1 & 2:
    Login as HOD via User ID or Email -> validates credentials -> HOD Main Dashboard.
    """
    c = flow_client["client"]

    # 1. Via User ID
    resp = c.post("/api/auth/login", json={"user_id": "HOD001", "password": "admin123"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["user"]["role"] == "hod"
    assert data["redirect_url"] == "/hod/dashboard"
    assert "access_token" in data

    # 2. Via Email
    resp_email = c.post("/api/auth/login", json={"user_id": "hod.aids@college.edu", "password": "admin123"})
    assert resp_email.status_code == 200
    data_email = resp_email.json()["data"]
    assert data_email["user"]["role"] == "hod"
    assert data_email["redirect_url"] == "/hod/dashboard"


def test_03_authenticate_advisor_by_user_id_and_email(flow_client):
    """
    MASTER FLOW 1 & 4:
    Login as Advisor via User ID or Email -> validates credentials -> Advisor Dashboard.
    """
    c = flow_client["client"]

    # 1. Via User ID
    resp = c.post("/api/auth/login", json={"user_id": "ADV001", "password": "password123"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["user"]["role"] == "class_advisor"
    assert data["redirect_url"] == "/advisor/dashboard"

    # 2. Via Email
    resp_email = c.post("/api/auth/login", json={"user_id": "ramesh@college.edu", "password": "password123"})
    assert resp_email.status_code == 200
    data_email = resp_email.json()["data"]
    assert data_email["user"]["role"] == "class_advisor"
    assert data_email["redirect_url"] == "/advisor/dashboard"


def test_04_advisor_flow_unassigned_class(flow_client):
    """
    MASTER FLOW 4:
    Advisor with NO CLASS ASSIGNED -> status_code: NO_CLASS_ASSIGNED.
    """
    c = flow_client["client"]
    user_svc = flow_client["user_service"]
    adv2 = user_svc.get_user_by_id("ADV002")
    token = create_access_token(user_id=adv2.user_id, role="class_advisor", name=adv2.name)

    resp = c.get("/api/advisor/classroom-status", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["status_code"] == "NO_CLASS_ASSIGNED"
    assert "No Class Assigned" in data["title"]


def test_05_advisor_flow_classroom_not_configured(flow_client):
    """
    MASTER FLOW 4:
    Class Assigned but Classroom NOT CONFIGURED -> status_code: NOT_CONFIGURED
    Title: CLASSROOM NOT CONFIGURED, Message: Camera unavailable.
    """
    c = flow_client["client"]
    user_svc = flow_client["user_service"]

    # Create Advisor with assigned class but nonexistent classroom
    adv3 = UserModel(
        user_id="ADV003",
        name="Prof. Unconfigured",
        role=UserRole.CLASS_ADVISOR,
        password_hash=hash_password("password123"),
        department="AI&DS",
        year="2nd Year",
        section="A",
        assigned_classroom="AIDS-NONEXISTENT",
        status=UserStatus.ACTIVE
    )
    user_svc.create_user(adv3)
    token = create_access_token(
        user_id=adv3.user_id,
        role="class_advisor",
        name=adv3.name,
        department=adv3.department,
        year=adv3.year,
        section=adv3.section,
        assigned_classroom=adv3.assigned_classroom
    )

    resp = c.get("/api/advisor/classroom-status", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["status_code"] == "NOT_CONFIGURED"
    assert "CLASSROOM NOT CONFIGURED" in data["title"]
    assert "Camera unavailable" in data["message"]


def test_06_advisor_flow_camera_offline(flow_client):
    """
    MASTER FLOW 4:
    Classroom Configured but Camera Offline -> status_code: CAMERA_OFFLINE.
    """
    c = flow_client["client"]
    user_svc = flow_client["user_service"]
    adv1 = user_svc.get_user_by_id("ADV001")
    token = create_access_token(
        user_id=adv1.user_id,
        role="class_advisor",
        name=adv1.name,
        department=adv1.department,
        year=adv1.year,
        section=adv1.section,
        assigned_classroom=adv1.assigned_classroom
    )

    resp = c.get("/api/advisor/classroom-status", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    # Either CAMERA_OFFLINE or ONLINE depending on physical camera state
    assert data["status_code"] in ("CAMERA_OFFLINE", "ONLINE")
    if data["status_code"] == "CAMERA_OFFLINE":
        assert "CAMERA OFFLINE" in data["title"]


def test_07_hod_global_access_classrooms_and_crud(flow_client):
    """
    MASTER FLOW 2 & 3:
    HOD has Global Access:
    - View Classrooms
    - Add Classroom
    - Edit Classroom
    - Delete Classroom
    """
    c = flow_client["client"]
    user_svc = flow_client["user_service"]
    hod = user_svc.get_user_by_id("HOD001")
    token = create_access_token(user_id=hod.user_id, role="hod", name=hod.name)

    # 1. View Classrooms
    resp_get = c.get("/api/classrooms", headers={"Authorization": f"Bearer {token}"})
    assert resp_get.status_code == 200
    assert resp_get.json()["success"] is True

    # 2. Add Classroom
    new_room = {
        "classroom_id": "TEST-ROOM-101",
        "classroom_name": "Test Lab 101",
        "department": "AI&DS",
        "year": "4th Year",
        "section": "A",
        "assigned_advisor_id": "ADV001",
        "camera_source": "pc",
        "esp32_device_id": "ESP32-101"
    }
    resp_add = c.post("/api/hod/classrooms", json=new_room, headers={"Authorization": f"Bearer {token}"})
    assert resp_add.status_code == 201
    assert resp_add.json()["data"]["classroom_id"] == "TEST-ROOM-101"

    # 3. Edit Classroom
    update_data = {
        "classroom_name": "Test Lab 101 Updated",
        "camera_source": "droidcam"
    }
    resp_put = c.put("/api/hod/classrooms/TEST-ROOM-101", json=update_data, headers={"Authorization": f"Bearer {token}"})
    assert resp_put.status_code == 200
    assert resp_put.json()["data"]["classroom_name"] == "Test Lab 101 Updated"
    assert resp_put.json()["data"]["camera_source"] == "droidcam"

    # 4. Delete Classroom
    resp_del = c.delete("/api/hod/classrooms/TEST-ROOM-101", headers={"Authorization": f"Bearer {token}"})
    assert resp_del.status_code == 200
    assert resp_del.json()["success"] is True
