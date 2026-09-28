import os
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.user_service import UserService
from app.models.user import UserModel, UserRole, UserStatus
from app.core.security import hash_password, create_access_token


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Provides a TestClient with isolated temporary user database."""
    temp_dir = tmp_path_factory.mktemp("rbac_test")
    test_db = str(temp_dir / "smartclass.sqlite")
    
    # Initialize UserService with temporary test DB
    user_svc = UserService(db_path=test_db, mongo_uri="mongodb://localhost:27017")
    
    # Override user service in app state / dependency
    app = create_app()
    from app.auth import get_user_service
    app.dependency_overrides[get_user_service] = lambda: user_svc

    with TestClient(app) as test_client:
        yield {
            "client": test_client,
            "user_service": user_svc,
            "db_path": test_db
        }


def _err_msg(resp) -> str:
    body = resp.json()
    if isinstance(body, dict):
        if "error" in body and body["error"]:
            return body["error"].get("message", "")
        if "detail" in body:
            return body.get("detail", "")
    return str(body)


def _get_hod_token(user_svc: UserService) -> str:
    hod = user_svc.get_user_by_id("HOD001")
    return create_access_token(
        user_id=hod.user_id,
        role=hod.role.value,
        name=hod.name,
        department=hod.department
    )


def _get_advisor_token(user_svc: UserService, user_id: str) -> str:
    adv = user_svc.get_user_by_id(user_id)
    return create_access_token(
        user_id=adv.user_id,
        role=adv.role.value,
        name=adv.name,
        department=adv.department,
        year=adv.year,
        section=adv.section,
        assigned_classroom=adv.assigned_classroom
    )


def test_01_hod_can_access_all_classes(client):
    """TEST 1: HOD can access all classes."""
    c = client["client"]
    token = _get_hod_token(client["user_service"])

    resp = c.get("/api/scoped/students", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert isinstance(data["data"], list)


def test_02_hod_can_create_advisor(client):
    """TEST 2: HOD can create Advisor."""
    c = client["client"]
    token = _get_hod_token(client["user_service"])

    payload = {
        "user_id": "ADV001",
        "name": "Dr. K. Rajesh",
        "password": "password123",
        "confirm_password": "password123",
        "department": "AI&DS",
        "year": "3rd Year",
        "section": "B",
        "assigned_classroom": "AIDS-B",
        "email": "rajesh@college.edu"
    }

    resp = c.post("/api/hod/advisors", json=payload, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["success"] is True
    assert data["data"]["user_id"] == "ADV001"
    assert data["data"]["role"] == "class_advisor"
    assert data["data"]["section"] == "B"


def test_03_hod_can_edit_advisor(client):
    """TEST 3: HOD can edit Advisor."""
    c = client["client"]
    token = _get_hod_token(client["user_service"])

    update_payload = {
        "name": "Dr. K. Rajesh Updated",
        "phone": "9876543210"
    }

    resp = c.put("/api/hod/advisors/ADV001", json=update_payload, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["data"]["name"] == "Dr. K. Rajesh Updated"
    assert data["data"]["phone"] == "9876543210"


def test_04_hod_can_disable_and_enable_advisor(client):
    """TEST 4: HOD can disable and re-enable Advisor."""
    c = client["client"]
    token = _get_hod_token(client["user_service"])

    # Disable
    resp = c.patch("/api/hod/advisors/ADV001/status?status=disabled", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["data"]["status"] == "disabled"

    adv = client["user_service"].get_user_by_id("ADV001")
    assert adv.status == UserStatus.DISABLED

    # Re-enable
    resp2 = c.patch("/api/hod/advisors/ADV001/status?status=active", headers={"Authorization": f"Bearer {token}"})
    assert resp2.status_code == 200
    assert resp2.json()["data"]["status"] == "active"


def test_05_advisor_can_login(client):
    """TEST 5: Advisor can login and receive JWT token with target dashboard."""
    c = client["client"]

    login_payload = {
        "user_id": "ADV001",
        "password": "password123"
    }
    resp = c.post("/api/auth/login", json=login_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "access_token" in data["data"]
    assert data["data"]["redirect_url"] == "/advisor/dashboard"
    assert data["data"]["user"]["role"] == "class_advisor"


def test_06_advisor_can_view_assigned_class(client):
    """TEST 6: Advisor can view assigned class summary."""
    c = client["client"]
    token = _get_advisor_token(client["user_service"], "ADV001")

    resp = c.get("/api/dashboard/role-summary", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["role"] == "class_advisor"
    assert data["department"] == "AI&DS"
    assert data["section"] == "B"
    assert data["assigned_classroom"] == "AIDS-B"
    assert "cards" in data
    assert "total_students" in data["cards"]


def test_07_advisor_can_view_assigned_students(client):
    """TEST 7: Advisor can view assigned students."""
    c = client["client"]
    token = _get_advisor_token(client["user_service"], "ADV001")

    resp = c.get("/api/scoped/students", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    # Students returned should only belong to section B
    for student in data["data"]:
        assert student["section"].upper() == "B"


def test_08_advisor_can_view_assigned_attendance(client):
    """TEST 8: Advisor can view assigned attendance."""
    c = client["client"]
    token = _get_advisor_token(client["user_service"], "ADV001")

    resp = c.get("/api/dashboard/role-summary", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    cards = resp.json()["data"]["cards"]
    assert "present_today" in cards
    assert "absent_today" in cards
    assert "attendance_percentage" in cards


def test_09_advisor_cannot_view_another_class(client):
    """TEST 9: Advisor cannot view another class (forbidden if attempting to tamper params)."""
    c = client["client"]
    token = _get_advisor_token(client["user_service"], "ADV001")

    # ADV001 is assigned to Section B. Tampering with Section A:
    resp = c.get("/api/scoped/students?section=A", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403
    assert "only authorized to access section 'B'" in _err_msg(resp)

    # Tampering with Department ECE:
    resp2 = c.get("/api/scoped/students?department=ECE", headers={"Authorization": f"Bearer {token}"})
    assert resp2.status_code == 403
    assert "only authorized to access department 'AI&DS'" in _err_msg(resp2)


def test_10_advisor_cannot_create_advisor(client):
    """TEST 10: Advisor cannot create Advisor."""
    c = client["client"]
    token = _get_advisor_token(client["user_service"], "ADV001")

    payload = {
        "user_id": "ADV999",
        "name": "Rogue Advisor",
        "password": "password123",
        "department": "AI&DS",
        "year": "3rd Year",
        "section": "A"
    }

    resp = c.post("/api/hod/advisors", json=payload, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403
    assert "HOD role required" in _err_msg(resp)


def test_11_advisor_cannot_edit_advisor(client):
    """TEST 11: Advisor cannot edit Advisor."""
    c = client["client"]
    token = _get_advisor_token(client["user_service"], "ADV001")

    resp = c.put("/api/hod/advisors/ADV001", json={"name": "Hacked Name"}, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403
    assert "HOD role required" in _err_msg(resp)


def test_12_advisor_cannot_access_hod_dashboard(client):
    """TEST 12: Advisor cannot access HOD advisor list endpoint."""
    c = client["client"]
    token = _get_advisor_token(client["user_service"], "ADV001")

    resp = c.get("/api/hod/advisors", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403
    assert "HOD role required" in _err_msg(resp)


def test_13_advisor_cannot_access_system_settings(client):
    """TEST 13: Advisor cannot reset another advisor's password."""
    c = client["client"]
    token = _get_advisor_token(client["user_service"], "ADV001")

    resp = c.post("/api/hod/advisors/ADV001/reset-password", json={"new_password": "newpass123"}, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403
    assert "HOD role required" in _err_msg(resp)


def test_14_disabled_advisor_cannot_login(client):
    """TEST 14: Disabled Advisor cannot login."""
    c = client["client"]
    user_svc = client["user_service"]

    try:
        # Disable ADV001
        user_svc.update_status("ADV001", "disabled")

        login_payload = {
            "user_id": "ADV001",
            "password": "password123"
        }
        resp = c.post("/api/auth/login", json=login_payload)
        assert resp.status_code == 403
        assert "disabled" in _err_msg(resp).lower()
    finally:
        # Guarantee re-enable for subsequent tests
        user_svc.update_status("ADV001", "active")


def test_15_duplicate_advisor_user_id_rejected(client):
    """TEST 15: Duplicate Advisor user ID is rejected."""
    c = client["client"]
    token = _get_hod_token(client["user_service"])

    payload = {
        "user_id": "ADV001",  # already exists
        "name": "Duplicate Advisor",
        "password": "password123",
        "department": "AI&DS",
        "year": "3rd Year",
        "section": "B"
    }

    resp = c.post("/api/hod/advisors", json=payload, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 409
    assert "already exists" in _err_msg(resp)


def test_16_advisor_cannot_bypass_class_restriction_through_api_params(client):
    """TEST 16: Advisor cannot bypass class restriction through API parameters."""
    c = client["client"]
    token = _get_advisor_token(client["user_service"], "ADV001")

    # Attempt to bypass by injecting different section / department
    for test_param in ["?section=C", "?department=MECH", "?section=A&department=CSE"]:
        resp = c.get(f"/api/scoped/students{test_param}", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403


def test_17_acceptance_scenario_hod_and_adv001(client):
    """
    TEST 17 (ACCEPTANCE CRITERIA):
    HOD creates ADV001 (3rd Year, AI&DS, Section B).
    ADV001 logs in -> sees ONLY 3rd Year AI&DS B.
    HOD logs in -> sees ALL classes, students, advisors, system status.
    """
    c = client["client"]
    user_svc = client["user_service"]

    # Ensure ADV001 is active
    user_svc.update_status("ADV001", "active")

    # 1. HOD creates advisor ADV002 (3rd Year, AI&DS, Section A) for multi-class comparison
    hod_token = _get_hod_token(user_svc)
    c.post("/api/hod/advisors", json={
        "user_id": "ADV002",
        "name": "Prof. S. Anita",
        "password": "password456",
        "department": "AI&DS",
        "year": "3rd Year",
        "section": "A",
        "assigned_classroom": "AIDS-A"
    }, headers={"Authorization": f"Bearer {hod_token}"})

    # 2. ADV001 logs in
    adv1_login = c.post("/api/auth/login", json={"user_id": "ADV001", "password": "password123"})
    assert adv1_login.status_code == 200
    adv1_token = adv1_login.json()["data"]["access_token"]
    assert adv1_login.json()["data"]["redirect_url"] == "/advisor/dashboard"

    # 3. ADV001 checks scoped students
    adv1_students = c.get("/api/scoped/students", headers={"Authorization": f"Bearer {adv1_token}"}).json()["data"]
    # All students visible to ADV001 must belong to section B
    for s in adv1_students:
        assert s["section"].upper() == "B"

    # ADV001 cannot view advisors
    adv1_advisors = c.get("/api/hod/advisors", headers={"Authorization": f"Bearer {adv1_token}"})
    assert adv1_advisors.status_code == 403

    # 4. HOD logs in
    hod_login = c.post("/api/auth/login", json={"user_id": "HOD001", "password": "admin123"})
    assert hod_login.status_code == 200
    hod_token_login = hod_login.json()["data"]["access_token"]
    assert hod_login.json()["data"]["redirect_url"] == "/hod/dashboard"

    # HOD can see ALL advisors
    hod_advisors = c.get("/api/hod/advisors", headers={"Authorization": f"Bearer {hod_token_login}"}).json()["data"]
    adv_ids = [a["user_id"] for a in hod_advisors]
    assert "ADV001" in adv_ids
    assert "ADV002" in adv_ids

    # HOD can see full system status
    hod_sum = c.get("/api/dashboard/role-summary", headers={"Authorization": f"Bearer {hod_token_login}"}).json()["data"]
    assert hod_sum["role"] == "hod"
    assert "system_status" in hod_sum
    assert hod_sum["system_status"]["fastapi"] == "ONLINE"
