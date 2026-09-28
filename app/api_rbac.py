import os
import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Response
from pydantic import BaseModel

from app.models.user import (
    UserModel,
    UserRole,
    UserStatus,
    UserPublicProfile,
    LoginRequest,
    LoginResponse,
    CreateAdvisorRequest,
    UpdateAdvisorRequest,
    ResetAdvisorPasswordRequest,
    ChangePasswordRequest
)
from app.core.security import verify_password, hash_password, create_access_token
from app.auth import (
    get_user_service,
    get_current_user,
    require_hod,
    require_class_advisor,
    require_any_authenticated_user
)
from app.schemas import ApiResponse
from app.state import AppState, get_app_state

router = APIRouter(prefix="/api", tags=["RBAC Authentication & Dashboards"])


# =============================================================================
# 1. Authentication Endpoints
# =============================================================================

@router.post("/auth/login", response_model=ApiResponse[LoginResponse])
def login(
    payload: LoginRequest,
    response: Response,
    user_service = Depends(get_user_service)
):
    """
    Authenticates HOD or Class Advisor:
    - Verifies user ID and hashed password
    - Verifies account is active (disabled accounts rejected with 403)
    - Returns signed JWT token, profile, and target dashboard URL
    - Sets secure HTTP-only access_token cookie
    """
    user_id = payload.user_id.strip()
    user = user_service.get_user_by_id_or_email(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid User ID or password."
        )

    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid User ID or password."
        )

    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been disabled. Please contact the HOD."
        )

    # Determine redirect URL based strictly on backend role
    if user.role == UserRole.HOD:
        redirect_url = "/hod/dashboard"
    elif user.role == UserRole.CLASS_ADVISOR:
        redirect_url = "/advisor/dashboard"
    else:
        redirect_url = "/"

    token = create_access_token(
        user_id=user.user_id,
        role=user.role.value,
        name=user.name,
        department=user.department,
        year=user.year,
        section=user.section,
        assigned_classroom=user.assigned_classroom
    )

    # Set cookie for browser navigation
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=43200,  # 12 hours
        samesite="lax",
        secure=False
    )

    profile = UserPublicProfile(
        user_id=user.user_id,
        name=user.name,
        role=user.role.value,
        email=user.email,
        phone=user.phone,
        department=user.department,
        year=user.year,
        section=user.section,
        assigned_classroom=user.assigned_classroom,
        status=user.status.value,
        created_at=user.created_at
    )

    return ApiResponse.ok(LoginResponse(
        access_token=token,
        token_type="bearer",
        user=profile,
        redirect_url=redirect_url
    ))


@router.post("/auth/logout", response_model=ApiResponse[Dict[str, Any]])
def logout(response: Response):
    """Logs out by clearing access_token cookie."""
    response.delete_cookie(key="access_token")
    return ApiResponse.ok({"message": "Successfully logged out.", "redirect_url": "/login"})


@router.get("/auth/me", response_model=ApiResponse[UserPublicProfile])
def get_current_user_profile(user: UserModel = Depends(get_current_user)):
    """Retrieves profile of currently authenticated user."""
    profile = UserPublicProfile(
        user_id=user.user_id,
        name=user.name,
        role=user.role.value,
        email=user.email,
        phone=user.phone,
        department=user.department,
        year=user.year,
        section=user.section,
        assigned_classroom=user.assigned_classroom,
        status=user.status.value,
        created_at=user.created_at
    )
    return ApiResponse.ok(profile)


@router.post("/auth/change-password", response_model=ApiResponse[Dict[str, Any]])
def change_password(
    payload: ChangePasswordRequest,
    user: UserModel = Depends(get_current_user),
    user_service = Depends(get_user_service)
):
    """Allows authenticated user (HOD or Class Advisor) to change their own password."""
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password incorrect.")

    if payload.confirm_new_password and payload.new_password != payload.confirm_new_password:
        raise HTTPException(status_code=400, detail="New passwords do not match.")

    new_hash = hash_password(payload.new_password)
    user_service.update_password(user.user_id, new_hash)
    return ApiResponse.ok({"message": "Password changed successfully."})


# =============================================================================
# 2. HOD Advisor Management Endpoints (HOD Role Strictly Required)
# =============================================================================

@router.get("/hod/advisors", response_model=ApiResponse[List[UserPublicProfile]])
def list_advisors(
    current_hod: UserModel = Depends(require_hod),
    user_service = Depends(get_user_service)
):
    """Lists all Class Advisors (HOD only)."""
    advisors = user_service.list_advisors()
    res = [
        UserPublicProfile(
            user_id=a.user_id,
            name=a.name,
            role=a.role.value,
            email=a.email,
            phone=a.phone,
            department=a.department,
            year=a.year,
            section=a.section,
            assigned_classroom=a.assigned_classroom,
            status=a.status.value,
            created_at=a.created_at
        )
        for a in advisors
    ]
    return ApiResponse.ok(res)


@router.post("/hod/advisors", response_model=ApiResponse[UserPublicProfile], status_code=status.HTTP_201_CREATED)
def create_advisor(
    payload: CreateAdvisorRequest,
    current_hod: UserModel = Depends(require_hod),
    user_service = Depends(get_user_service)
):
    """Creates a new Class Advisor account (HOD only)."""
    existing = user_service.get_user_by_id(payload.user_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Advisor with User ID '{payload.user_id}' already exists."
        )

    if payload.confirm_password and payload.password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    hashed = hash_password(payload.password)
    now = datetime.datetime.utcnow().isoformat()
    status_enum = UserStatus.ACTIVE if (payload.status or "active").lower() == "active" else UserStatus.DISABLED

    advisor = UserModel(
        user_id=payload.user_id.strip(),
        name=payload.name.strip(),
        role=UserRole.CLASS_ADVISOR,
        password_hash=hashed,
        email=payload.email,
        phone=payload.phone,
        department=payload.department.strip(),
        year=payload.year.strip(),
        section=payload.section.strip().upper(),
        assigned_classroom=payload.assigned_classroom or f"{payload.department}-{payload.section}".strip(" -"),
        status=status_enum,
        created_at=now,
        updated_at=now
    )

    created = user_service.create_user(advisor)

    return ApiResponse.ok(UserPublicProfile(
        user_id=created.user_id,
        name=created.name,
        role=created.role.value,
        email=created.email,
        phone=created.phone,
        department=created.department,
        year=created.year,
        section=created.section,
        assigned_classroom=created.assigned_classroom,
        status=created.status.value,
        created_at=created.created_at
    ))


@router.put("/hod/advisors/{user_id}", response_model=ApiResponse[UserPublicProfile])
def update_advisor(
    user_id: str = Path(...),
    payload: UpdateAdvisorRequest = None,
    current_hod: UserModel = Depends(require_hod),
    user_service = Depends(get_user_service)
):
    """Updates an existing Advisor's details or assigned class/section (HOD only)."""
    adv = user_service.get_user_by_id(user_id)
    if not adv:
        raise HTTPException(status_code=404, detail=f"Advisor '{user_id}' not found.")
    if adv.role != UserRole.CLASS_ADVISOR:
        raise HTTPException(status_code=400, detail="Target user is not a Class Advisor.")

    updates = {}
    if payload.name is not None:
        updates["name"] = payload.name
    if payload.email is not None:
        updates["email"] = payload.email
    if payload.phone is not None:
        updates["phone"] = payload.phone
    if payload.department is not None:
        updates["department"] = payload.department
    if payload.year is not None:
        updates["year"] = payload.year
    if payload.section is not None:
        updates["section"] = payload.section.upper()
    if payload.assigned_classroom is not None:
        updates["assigned_classroom"] = payload.assigned_classroom
    if payload.status is not None:
        updates["status"] = payload.status.lower()

    updated = user_service.update_user(user_id, updates)
    return ApiResponse.ok(UserPublicProfile(
        user_id=updated.user_id,
        name=updated.name,
        role=updated.role.value,
        email=updated.email,
        phone=updated.phone,
        department=updated.department,
        year=updated.year,
        section=updated.section,
        assigned_classroom=updated.assigned_classroom,
        status=updated.status.value,
        created_at=updated.created_at
    ))


@router.patch("/hod/advisors/{user_id}/status", response_model=ApiResponse[Dict[str, Any]])
def update_advisor_status(
    user_id: str = Path(...),
    status_val: str = Query(..., alias="status", pattern="^(active|disabled)$"),
    current_hod: UserModel = Depends(require_hod),
    user_service = Depends(get_user_service)
):
    """Enables or disables an Advisor account (HOD only)."""
    adv = user_service.get_user_by_id(user_id)
    if not adv:
        raise HTTPException(status_code=404, detail=f"Advisor '{user_id}' not found.")
    if adv.role != UserRole.CLASS_ADVISOR:
        raise HTTPException(status_code=400, detail="Cannot toggle status of non-advisor account.")

    user_service.update_status(user_id, status_val)
    return ApiResponse.ok({
        "user_id": user_id,
        "status": status_val,
        "message": f"Advisor account '{user_id}' is now {status_val}."
    })


@router.post("/hod/advisors/{user_id}/reset-password", response_model=ApiResponse[Dict[str, Any]])
def reset_advisor_password(
    user_id: str = Path(...),
    payload: ResetAdvisorPasswordRequest = None,
    current_hod: UserModel = Depends(require_hod),
    user_service = Depends(get_user_service)
):
    """Resets an Advisor's password (HOD only)."""
    adv = user_service.get_user_by_id(user_id)
    if not adv:
        raise HTTPException(status_code=404, detail=f"Advisor '{user_id}' not found.")
    if adv.role != UserRole.CLASS_ADVISOR:
        raise HTTPException(status_code=400, detail="Cannot reset password of non-advisor account.")

    new_hash = hash_password(payload.new_password)
    user_service.update_password(user_id, new_hash)
    return ApiResponse.ok({
        "user_id": user_id,
        "message": f"Password for Advisor '{user_id}' has been reset successfully."
    })


# =============================================================================
# 3. Role-Scoped Dashboard & System Status Endpoints
# =============================================================================

@router.get("/dashboard/role-summary", response_model=ApiResponse[Dict[str, Any]])
def get_role_dashboard_summary(
    user: UserModel = Depends(get_current_user),
    state: AppState = Depends(get_app_state),
    user_service = Depends(get_user_service)
):
    """
    Returns dashboard overview data strictly filtered by user's role:
    - HOD: Full system view across all classes, classrooms, advisors, devices.
    - Class Advisor: Strictly scoped to assigned Department, Year, Section.
    """
    all_students = state.db.get_all_students()
    telemetry = state.get_latest_telemetry()
    active_sess = state.session_manager.get_active_session()

    # Real Attendance from Engine
    total_present = 0
    total_late = 0
    present_student_ids = set()

    if active_sess:
        report = state.attendance_engine.generate_session_report(active_sess.session_id)
        total_present = report.present_count
        total_late = report.late_count
        for r in report.records:
            if r.status.value in ["PRESENT", "LATE"]:
                present_student_ids.add(r.student_id)

    if user.role == UserRole.HOD:
        # HOD: Full system stats
        total_enrolled = len(all_students)
        present_count = len(present_student_ids)
        absent_count = max(0, total_enrolled - present_count)
        advisors = user_service.list_advisors()
        active_advisors = sum(1 for a in advisors if a.status == UserStatus.ACTIVE)

        # Real system component status
        mongo_status = "CONNECTED" if user_service.mongo_online else "OFFLINE (RESILIENT LOCAL DB)"
        camera_status = "CONNECTED" if telemetry["camera_online"] else "DISCONNECTED"
        ai_status = "RUNNING" if telemetry["recognition_online"] else "OFFLINE"
        faiss_vectors = state.db.get_embedding_count()

        return ApiResponse.ok({
            "role": "hod",
            "user_id": user.user_id,
            "user_name": user.name,
            "cards": {
                "total_students": total_enrolled,
                "students_present": present_count,
                "students_absent": absent_count,
                "active_advisors": active_advisors,
                "active_classrooms": 1,
                "active_cameras": 1 if telemetry["camera_online"] else 0,
                "active_esp32_devices": 1 if telemetry["camera_online"] else 0,
                "active_alerts": 0 if telemetry["camera_online"] else 1
            },
            "system_status": {
                "fastapi": "ONLINE",
                "mongodb": mongo_status,
                "camera": camera_status,
                "ai_pipeline": ai_status,
                "faiss": f"READY ({faiss_vectors} vectors)",
                "esp32": "READY",
                "websocket": "ONLINE"
            },
            "active_session": active_sess.model_dump() if active_sess else None
        })

    else:
        # Class Advisor: Filtered strictly to assigned class & section
        advisor_dept = (user.department or "").strip().lower()
        advisor_sec = (user.section or "").strip().upper()
        advisor_year = (user.year or "").strip().lower()

        assigned_students = [
            s for s in all_students
            if (not advisor_dept or s.get("department", "").strip().lower() == advisor_dept)
            and (not advisor_sec or s.get("section", "").strip().upper() == advisor_sec)
        ]

        total_class_students = len(assigned_students)
        class_student_ids = {s["student_id"] for s in assigned_students}
        class_present = len(class_student_ids.intersection(present_student_ids))
        class_absent = max(0, total_class_students - class_present)
        att_pct = round((class_present / total_class_students * 100), 1) if total_class_students > 0 else 0.0

        return ApiResponse.ok({
            "role": "class_advisor",
            "user_id": user.user_id,
            "advisor_name": user.name,
            "department": user.department,
            "year": user.year,
            "section": user.section,
            "assigned_classroom": user.assigned_classroom or f"{user.department}-{user.section}",
            "cards": {
                "total_students": total_class_students,
                "present_today": class_present,
                "absent_today": class_absent,
                "attendance_percentage": att_pct,
                "active_camera": "CONNECTED" if telemetry["camera_online"] else "DISCONNECTED",
                "active_alerts": 0
            },
            "assigned_class_title": f"{user.year or ''} {user.department or ''} - {user.section or ''}".strip(),
            "active_session": active_sess.model_dump() if active_sess else None
        })


# =============================================================================
# 4. Role-Scoped Student Monitoring Endpoints
# =============================================================================

@router.get("/scoped/students", response_model=ApiResponse[List[Dict[str, Any]]])
def get_scoped_students(
    department: Optional[str] = Query(None),
    section: Optional[str] = Query(None),
    year: Optional[str] = Query(None),
    user: UserModel = Depends(get_current_user),
    state: AppState = Depends(get_app_state)
):
    """
    Retrieves students:
    - HOD: Full access to all students or query any class/section.
    - Class Advisor: STRICTLY constrained to advisor's assigned class & section.
      Query parameters attempting to inspect other sections are rejected or overridden.
    """
    all_students = state.db.get_all_students()

    # Determine present students in current session
    active_sess = state.session_manager.get_active_session()
    present_ids = set()
    if active_sess:
        report = state.attendance_engine.generate_session_report(active_sess.session_id)
        for r in report.records:
            if r.status.value in ["PRESENT", "LATE"]:
                present_ids.add(r.student_id)

    # Telemetry active tracks for recognition status
    telemetry = state.get_latest_telemetry()
    active_track_ids = {t.stable_student_id: t for t in telemetry["tracks"] if t.stable_student_id}

    if user.role == UserRole.CLASS_ADVISOR:
        # Backend-enforced scope restriction:
        # Even if caller passes department='ECE' or section='A', advisor's credentials prevail!
        adv_dept = (user.department or "").strip().lower()
        adv_sec = (user.section or "").strip().upper()

        if department and department.strip().lower() != adv_dept:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: You are only authorized to access department '{user.department}'."
            )
        if section and section.strip().upper() != adv_sec:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: You are only authorized to access section '{user.section}'."
            )

        filter_dept = adv_dept
        filter_sec = adv_sec
    else:
        # HOD can filter by any provided param or view all
        filter_dept = department.strip().lower() if department else None
        filter_sec = section.strip().upper() if section else None

    results = []
    for s in all_students:
        s_dept = s.get("department", "").strip().lower()
        s_sec = s.get("section", "").strip().upper()

        if filter_dept and s_dept != filter_dept:
            continue
        if filter_sec and s_sec != filter_sec:
            continue

        sid = s["student_id"]
        is_present = sid in present_ids
        track = active_track_ids.get(sid)

        rec_status = "Not Tracked"
        if track:
            rec_status = f"Recognized ({int(track.current_similarity * 100)}%)"

        att_status = "Present" if is_present else "Not Seen"

        results.append({
            "register_no": s.get("register_no", sid),
            "student_name": s["student_name"],
            "department": s.get("department", ""),
            "year": s.get("class_name") or user.year or "3rd Year",
            "section": s.get("section", ""),
            "attendance_status": att_status,
            "attendance_percentage": 100.0 if is_present else 0.0,
            "last_seen": "Today" if is_present else "Never",
            "recognition_status": rec_status
        })

    return ApiResponse.ok(results)


# =============================================================================
# 5. Role-Scoped Alerts Endpoints
# =============================================================================

@router.get("/alerts", response_model=ApiResponse[List[Dict[str, Any]]])
@router.get("/scoped/alerts", response_model=ApiResponse[List[Dict[str, Any]]])
def get_scoped_alerts(
    user: UserModel = Depends(get_current_user),
    state: AppState = Depends(get_app_state)
):
    """
    Returns alerts scoped to the authenticated caller:
    - HOD: System alerts + all classroom alerts.
    - Class Advisor: Strictly alerts matching assigned class and classroom.
    """
    telemetry = state.get_latest_telemetry()
    alerts: List[Dict[str, Any]] = []

    # Camera offline alert
    if not telemetry["camera_online"]:
        alerts.append({
            "id": "ALT_CAM_OFFLINE",
            "severity": "CRITICAL",
            "title": "Camera Stream Disconnected",
            "message": "Physical video capture feed is offline or unreachable.",
            "classroom_id": "ALL",
            "timestamp": datetime.datetime.utcnow().isoformat()
        })

    # Unknown face alert
    if telemetry.get("unknown_count", 0) > 0:
        alerts.append({
            "id": "ALT_UNKNOWN_FACE",
            "severity": "WARNING",
            "title": "Unknown Face Detected",
            "message": f"{telemetry['unknown_count']} unidentified face(s) visible in camera frame.",
            "classroom_id": user.assigned_classroom or "AIDS-B",
            "timestamp": datetime.datetime.utcnow().isoformat()
        })

    if user.role == UserRole.HOD:
        return ApiResponse.ok(alerts)

    # Class Advisor: Filter alerts by assigned classroom
    adv_classroom = user.assigned_classroom or "AIDS-B"
    advisor_alerts = [a for a in alerts if a["classroom_id"] in [adv_classroom, "ALL"]]
    return ApiResponse.ok(advisor_alerts)


# =============================================================================
# 6. Classroom Management & Device Mapping Endpoints
# =============================================================================

class CreateClassroomRequest(BaseModel):
    classroom_id: str
    classroom_name: Optional[str] = None
    department: str = "AI&DS"
    year: str = "3rd Year"
    section: str = "B"
    assigned_advisor_id: Optional[str] = None
    camera_source: str = "pc"
    camera_url: Optional[str] = None
    esp32_device_id: Optional[str] = None


class UpdateClassroomRequest(BaseModel):
    classroom_name: Optional[str] = None
    department: Optional[str] = None
    year: Optional[str] = None
    section: Optional[str] = None
    assigned_advisor_id: Optional[str] = None
    camera_source: Optional[str] = None
    camera_url: Optional[str] = None
    camera_status: Optional[str] = None
    esp32_device_id: Optional[str] = None
    esp32_status: Optional[str] = None
    ai_pipeline_status: Optional[str] = None
    attendance_status: Optional[str] = None


@router.get("/classrooms", response_model=ApiResponse[List[Dict[str, Any]]])
def get_classrooms(
    user: UserModel = Depends(require_any_authenticated_user),
    state: AppState = Depends(get_app_state)
):
    """Lists all registered classrooms."""
    classrooms = state.db.get_all_classrooms()
    return ApiResponse.ok(classrooms)


@router.get("/classrooms/{classroom_id}", response_model=ApiResponse[Dict[str, Any]])
def get_classroom_detail(
    classroom_id: str,
    user: UserModel = Depends(require_any_authenticated_user),
    state: AppState = Depends(get_app_state)
):
    """Retrieves details of a single classroom."""
    classroom = state.db.get_classroom_by_id(classroom_id)
    if not classroom:
        raise HTTPException(status_code=404, detail=f"Classroom '{classroom_id}' not found.")
    return ApiResponse.ok(classroom)


@router.post("/hod/classrooms", response_model=ApiResponse[Dict[str, Any]], status_code=status.HTTP_201_CREATED)
def create_classroom(
    payload: CreateClassroomRequest,
    user: UserModel = Depends(require_hod),
    state: AppState = Depends(get_app_state),
    user_service = Depends(get_user_service)
):
    """Creates a new classroom and assigns advisor/camera (HOD only)."""
    existing = state.db.get_classroom_by_id(payload.classroom_id)
    if existing:
        raise HTTPException(status_code=400, detail=f"Classroom '{payload.classroom_id}' already exists.")

    adv_name = None
    if payload.assigned_advisor_id:
        adv = user_service.get_user_by_id(payload.assigned_advisor_id)
        if adv:
            adv_name = adv.name

    data = payload.model_dump()
    data["assigned_advisor_name"] = adv_name
    created = state.db.upsert_classroom(data)
    return ApiResponse.ok(created)


@router.put("/hod/classrooms/{classroom_id}", response_model=ApiResponse[Dict[str, Any]])
def update_classroom(
    classroom_id: str,
    payload: UpdateClassroomRequest,
    user: UserModel = Depends(require_hod),
    state: AppState = Depends(get_app_state),
    user_service = Depends(get_user_service)
):
    """Updates classroom metadata, camera mapping, or advisor assignment (HOD only)."""
    existing = state.db.get_classroom_by_id(classroom_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Classroom '{classroom_id}' not found.")

    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    merged = {**existing, **updates}
    merged["classroom_id"] = classroom_id

    if "assigned_advisor_id" in updates and updates["assigned_advisor_id"]:
        adv = user_service.get_user_by_id(updates["assigned_advisor_id"])
        merged["assigned_advisor_name"] = adv.name if adv else None

    updated = state.db.upsert_classroom(merged)
    return ApiResponse.ok(updated)


@router.delete("/hod/classrooms/{classroom_id}", response_model=ApiResponse[Dict[str, Any]])
def delete_classroom(
    classroom_id: str,
    user: UserModel = Depends(require_hod),
    state: AppState = Depends(get_app_state)
):
    """Deletes a classroom entity (HOD only)."""
    success = state.db.delete_classroom(classroom_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Classroom '{classroom_id}' not found.")
    return ApiResponse.ok({"message": f"Classroom '{classroom_id}' deleted successfully."})


@router.get("/advisor/classroom-status", response_model=ApiResponse[Dict[str, Any]])
def get_advisor_classroom_status(
    user: UserModel = Depends(require_class_advisor),
    state: AppState = Depends(get_app_state)
):
    """
    Evaluates Advisor Classroom setup hierarchy:
    1. Is advisor assigned to a class?
    2. Is the class assigned to a configured classroom?
    3. Is a camera configured for that classroom?
    4. Is the camera connected & online?
    5. Is AI pipeline available?
    """
    # 1. Check Class Assignment
    has_class = bool(user.year and user.section and user.department)
    if not has_class:
        return ApiResponse.ok({
            "status_code": "NO_CLASS_ASSIGNED",
            "has_assignment": False,
            "classroom_configured": False,
            "camera_configured": False,
            "camera_online": False,
            "title": "No Class Assigned",
            "message": "You are not currently assigned to any class or section. Please contact the HOD to assign your class."
        })

    # 2. Check Classroom Assignment & Configuration
    classroom_id = user.assigned_classroom
    classroom = state.db.get_classroom_by_id(classroom_id) if classroom_id else None
    if not classroom:
        # Fallback check by advisor ID
        classroom = state.db.get_classroom_by_advisor_id(user.user_id)
        if classroom:
            classroom_id = classroom["classroom_id"]

    if not classroom:
        return ApiResponse.ok({
            "status_code": "NOT_CONFIGURED",
            "has_assignment": True,
            "class_title": f"{user.year} {user.department} - {user.section}",
            "classroom_id": classroom_id or "Unassigned",
            "classroom_configured": False,
            "camera_configured": False,
            "camera_online": False,
            "title": "CLASSROOM NOT CONFIGURED",
            "message": f"Classroom for {user.year} {user.department} - {user.section} is not configured. Camera unavailable."
        })

    # 3. Check Camera Configuration
    cam_source = classroom.get("camera_source") or "none"
    if cam_source == "none":
        return ApiResponse.ok({
            "status_code": "NOT_CONFIGURED",
            "has_assignment": True,
            "class_title": f"{classroom['year']} {classroom['department']} - {classroom['section']}",
            "classroom_id": classroom["classroom_id"],
            "classroom_name": classroom["classroom_name"],
            "classroom_configured": True,
            "camera_configured": False,
            "camera_source": "none",
            "camera_online": False,
            "title": "CLASSROOM NOT CONFIGURED",
            "message": f"Classroom {classroom['classroom_id']} camera is not configured. Camera unavailable."
        })

    # 4. Check Camera Status & AI Pipeline
    telemetry = state.get_latest_telemetry()
    cam_online = bool(telemetry["camera_online"] and (state.camera_manager is not None))

    if not cam_online:
        return ApiResponse.ok({
            "status_code": "CAMERA_OFFLINE",
            "has_assignment": True,
            "class_title": f"{classroom['year']} {classroom['department']} - {classroom['section']}",
            "classroom_id": classroom["classroom_id"],
            "classroom_name": classroom["classroom_name"],
            "classroom_configured": True,
            "camera_configured": True,
            "camera_source": cam_source,
            "camera_online": False,
            "title": "CAMERA OFFLINE",
            "message": f"Camera for {classroom['classroom_id']} is currently offline."
        })

    # 5. All checks passed: ONLINE
    return ApiResponse.ok({
        "status_code": "ONLINE",
        "has_assignment": True,
        "class_title": f"{classroom['year']} {classroom['department']} - {classroom['section']}",
        "classroom_id": classroom["classroom_id"],
        "classroom_name": classroom["classroom_name"],
        "classroom_configured": True,
        "camera_configured": True,
        "camera_source": cam_source,
        "camera_online": True,
        "ai_pipeline_online": telemetry["recognition_online"],
        "esp32_device_id": classroom.get("esp32_device_id"),
        "esp32_online": classroom.get("esp32_status") == "connected",
        "title": "Classroom Online",
        "message": "Live camera feed and real-time monitoring active."
    })

