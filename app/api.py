import io
import csv
import json
import time
import asyncio
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, Query, Path, HTTPException, status, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.schemas import (
    ApiResponse,
    PaginatedData,
    PaginationMeta,
    CreateSessionRequest,
    StartSessionRequest,
    EndSessionRequest,
    SystemHealthData,
    ActiveTrackItem,
    DashboardSummaryData,
    StudentProfileData,
    StudentDetailData,
    RecognitionStatusData,
    AttendanceExportReportJson,
    AttendanceExportSessionJson,
    AttendanceExportRecordJson,
    EnrollmentReportResponse,
    EnrollStudentRequest,
    SelectCameraRequest,
    TestCameraRequest
)
import base64
from enrollment.data_importer import StudentDataImporter, ImporterValidationError
from enrollment.validation_pipeline import EnrollmentStatus
from app.state import AppState, get_app_state
from app.auth import get_current_role, require_operator, AuthContext
from attendance.session_manager import (
    SessionManagerError,
    InvalidSessionTransitionError,
    DuplicateSessionError
)
from core.schemas import SessionState, AttendanceStatus, SessionInfo, SessionAttendanceReport
from config.settings import get_settings

router = APIRouter(prefix="/api", tags=["SmartClass Vision AI"])


# =============================================================================
# 1. Health & System Status
# =============================================================================

@router.get("/health", response_model=ApiResponse[SystemHealthData])
def get_health(state: AppState = Depends(get_app_state)):
    """System health across Camera, Detection, Recognition, Tracking, Attendance, Database, API."""
    health_data = state.get_system_health()
    return ApiResponse.ok(health_data)


@router.get("/recognition/status", response_model=ApiResponse[RecognitionStatusData])
def get_recognition_status(state: AppState = Depends(get_app_state)):
    """Recognition model metadata, FAISS index configuration, and operational status."""
    settings = get_settings().recognition
    total_vectors = 0
    try:
        total_vectors = state.db.get_embedding_count()
    except Exception:
        pass

    telemetry = state.get_latest_telemetry()
    status_str = "ONLINE" if telemetry["recognition_online"] else "OFFLINE"

    data = RecognitionStatusData(
        model_name=settings.model_name,
        embedding_dim=settings.embedding_dim,
        similarity_threshold=settings.similarity_threshold,
        device=settings.device,
        faiss_index_path=settings.index_path,
        index_total_vectors=total_vectors,
        status=status_str
    )
    return ApiResponse.ok(data)


# =============================================================================
# 2. Session Management Endpoints
# =============================================================================

@router.get("/session/current", response_model=ApiResponse[Optional[SessionInfo]])
def get_current_session(state: AppState = Depends(get_app_state)):
    """Retrieves the active instructional session, or null if no active session exists."""
    active_sess = state.session_manager.get_active_session()
    return ApiResponse.ok(active_sess)


@router.get("/sessions", response_model=ApiResponse[PaginatedData[SessionInfo]])
def list_sessions(
    page: int = Query(1, ge=1, description="Page number starting at 1"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    date: Optional[str] = Query(None, description="Filter by date YYYY-MM-DD"),
    status: Optional[str] = Query(None, description="Filter by SessionState"),
    state: AppState = Depends(get_app_state)
):
    """Retrieves paginated list of sessions with optional date and status filters."""
    if date:
        raw_sessions = state.db.get_sessions_by_date(date)
    else:
        raw_sessions = state.db.get_all_sessions()

    if status:
        status_upper = status.upper()
        raw_sessions = [s for s in raw_sessions if s.get("status") == status_upper]

    total_items = len(raw_sessions)
    total_pages = max(1, (total_items + page_size - 1) // page_size)

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    sliced = raw_sessions[start_idx:end_idx]

    items = [SessionInfo(**s) for s in sliced]
    return ApiResponse.ok(PaginatedData(
        items=items,
        pagination=PaginationMeta(
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages
        )
    ))


@router.get("/sessions/{session_id}", response_model=ApiResponse[SessionInfo])
def get_session(
    session_id: str = Path(..., description="Unique session ID"),
    state: AppState = Depends(get_app_state)
):
    """Retrieves specific session details by session_id."""
    sess = state.session_manager.get_session_info(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return ApiResponse.ok(sess)


@router.post("/sessions", response_model=ApiResponse[SessionInfo], status_code=status.HTTP_201_CREATED)
def create_session(
    payload: CreateSessionRequest,
    auth: AuthContext = Depends(require_operator),
    state: AppState = Depends(get_app_state)
):
    """Creates a new scheduled instructional session (Operator role required)."""
    try:
        sess = state.session_manager.create_session(
            session_id=payload.session_id,
            date=payload.date,
            class_section=payload.class_section,
            subject=payload.subject,
            planned_start_time=payload.planned_start_time,
            planned_end_time=payload.planned_end_time
        )
        return ApiResponse.ok(sess)
    except DuplicateSessionError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except SessionManagerError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/sessions/{session_id}/start", response_model=ApiResponse[SessionInfo])
def start_session(
    session_id: str = Path(...),
    payload: Optional[StartSessionRequest] = None,
    auth: AuthContext = Depends(require_operator),
    state: AppState = Depends(get_app_state)
):
    """Transitions a session to ACTIVE state (Operator role required)."""
    start_time = payload.actual_start_time if payload else None
    try:
        sess = state.session_manager.start_session(session_id=session_id, actual_start_time=start_time)
        return ApiResponse.ok(sess)
    except DuplicateSessionError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except InvalidSessionTransitionError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except SessionManagerError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/sessions/{session_id}/pause", response_model=ApiResponse[SessionInfo])
def pause_session(
    session_id: str = Path(...),
    auth: AuthContext = Depends(require_operator),
    state: AppState = Depends(get_app_state)
):
    """Pauses an ACTIVE session (Operator role required)."""
    try:
        sess = state.session_manager.pause_session(session_id)
        return ApiResponse.ok(sess)
    except InvalidSessionTransitionError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except SessionManagerError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/sessions/{session_id}/resume", response_model=ApiResponse[SessionInfo])
def resume_session(
    session_id: str = Path(...),
    auth: AuthContext = Depends(require_operator),
    state: AppState = Depends(get_app_state)
):
    """Resumes a PAUSED or RECOVERING session back to ACTIVE (Operator role required)."""
    try:
        sess = state.session_manager.resume_session(session_id)
        return ApiResponse.ok(sess)
    except InvalidSessionTransitionError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except SessionManagerError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/sessions/{session_id}/end", response_model=ApiResponse[SessionInfo])
def end_session(
    session_id: str = Path(...),
    payload: Optional[EndSessionRequest] = None,
    auth: AuthContext = Depends(require_operator),
    state: AppState = Depends(get_app_state)
):
    """Ends an ACTIVE or PAUSED session. Requires operator confirmation (Terminal State)."""
    if payload and not payload.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session termination requires explicit confirmation (confirm=true)."
        )

    end_time = payload.actual_end_time if payload else None
    try:
        sess = state.session_manager.end_session(session_id=session_id, actual_end_time=end_time)
        return ApiResponse.ok(sess)
    except InvalidSessionTransitionError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except SessionManagerError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# =============================================================================
# 3. Attendance Endpoints
# =============================================================================

@router.get("/attendance/current", response_model=ApiResponse[Optional[SessionAttendanceReport]])
def get_current_attendance(state: AppState = Depends(get_app_state)):
    """Retrieves full attendance report for the currently active session."""
    active_sess = state.session_manager.get_active_session()
    if not active_sess:
        return ApiResponse.ok(None)

    report = state.attendance_engine.generate_session_report(active_sess.session_id)
    return ApiResponse.ok(report)


@router.get("/attendance/session/{session_id}", response_model=ApiResponse[SessionAttendanceReport])
def get_session_attendance(
    session_id: str = Path(...),
    state: AppState = Depends(get_app_state)
):
    """Retrieves full attendance report for any specified session."""
    sess = state.session_manager.get_session_info(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    report = state.attendance_engine.generate_session_report(session_id)
    return ApiResponse.ok(report)


# =============================================================================
# 4. Student Directory Endpoints
# =============================================================================

@router.get("/students", response_model=ApiResponse[PaginatedData[StudentProfileData]])
def list_students(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    query: Optional[str] = Query(None, description="Search by name or student_id"),
    section: Optional[str] = Query(None, description="Filter by section"),
    state: AppState = Depends(get_app_state)
):
    """Lists registered students with search and section filtering."""
    raw_students = state.db.get_all_students()

    if section:
        sec_upper = section.upper()
        raw_students = [s for s in raw_students if s.get("section", "").upper() == sec_upper]

    if query:
        q_lower = query.lower().strip()
        raw_students = [
            s for s in raw_students
            if q_lower in s.get("student_id", "").lower() or q_lower in s.get("student_name", "").lower()
        ]

    total_items = len(raw_students)
    total_pages = max(1, (total_items + page_size - 1) // page_size)

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    sliced = raw_students[start_idx:end_idx]

    items = [
        StudentProfileData(
            student_id=s["student_id"],
            student_name=s["student_name"],
            department=s.get("department", "Computer Science"),
            section=s.get("section", "A"),
            status=s.get("status", "active"),
            created_at=s.get("created_at"),
            updated_at=s.get("updated_at")
        )
        for s in sliced
    ]

    return ApiResponse.ok(PaginatedData(
        items=items,
        pagination=PaginationMeta(
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages
        )
    ))


@router.get("/students/{student_id}", response_model=ApiResponse[StudentDetailData])
def get_student_detail(
    student_id: str = Path(...),
    state: AppState = Depends(get_app_state)
):
    """Retrieves student profile and full attendance history across all recorded sessions."""
    student = state.db.get_student(student_id)
    if not student:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' not found.")

    history = state.db.get_student_attendance_history(student_id)
    attended_count = sum(1 for h in history if h.get("status") in (AttendanceStatus.PRESENT, AttendanceStatus.LATE))

    detail = StudentDetailData(
        profile=StudentProfileData(
            student_id=student["student_id"],
            student_name=student["student_name"],
            department=student.get("department", "Computer Science"),
            section=student.get("section", "A"),
            status=student.get("status", "active"),
            created_at=student.get("created_at"),
            updated_at=student.get("updated_at")
        ),
        attendance_history=history,
        total_sessions_attended=attended_count
    )
    return ApiResponse.ok(detail)


# =============================================================================
# 5. Dashboard Summary & Active Tracks
# =============================================================================

@router.get("/tracks/active", response_model=ApiResponse[List[ActiveTrackItem]])
def get_active_tracks(state: AppState = Depends(get_app_state)):
    """Retrieves active face tracks with bounding box, stability, and recognition state."""
    telemetry = state.get_latest_telemetry()
    tracks: List[ActiveTrackItem] = []
    for t in telemetry["tracks"]:
        tracks.append(ActiveTrackItem(
            track_id=t.track_id,
            bbox=t.bbox,
            state=t.state,
            stable_student_id=t.stable_student_id,
            stable_student_name=t.stable_student_name,
            current_similarity=t.current_similarity,
            quality_status=t.quality_status,
            current_status=t.current_status,
            hits=t.hits,
            age=t.age
        ))
    return ApiResponse.ok(tracks)


@router.get("/dashboard/summary", response_model=ApiResponse[DashboardSummaryData])
def get_dashboard_summary(state: AppState = Depends(get_app_state)):
    """Aggregates all live classroom metrics for high-speed Smart Board rendering."""
    active_sess = state.session_manager.get_active_session()
    telemetry = state.get_latest_telemetry()

    active_tracks_list: List[ActiveTrackItem] = []
    for t in telemetry["tracks"]:
        active_tracks_list.append(ActiveTrackItem(
            track_id=t.track_id,
            bbox=t.bbox,
            state=t.state,
            stable_student_id=t.stable_student_id,
            stable_student_name=t.stable_student_name,
            current_similarity=t.current_similarity,
            quality_status=t.quality_status,
            current_status=t.current_status,
            hits=t.hits,
            age=t.age
        ))

    if active_sess:
        report = state.attendance_engine.generate_session_report(active_sess.session_id)
        session_info = report.session
        session_status = active_sess.status
        total_enrolled = report.total_enrolled
        present_count = report.present_count
        late_count = report.late_count
        not_seen_count = report.not_seen_count
    else:
        session_info = None
        session_status = "NO_ACTIVE_SESSION"
        total_enrolled = state.db.get_student_count()
        present_count = 0
        late_count = 0
        not_seen_count = total_enrolled

    system_status = "ONLINE" if (telemetry["camera_online"] and telemetry["recognition_online"]) else "DEGRADED"

    summary = DashboardSummaryData(
        session=session_info,
        session_status=session_status,
        total_enrolled=total_enrolled,
        present_count=present_count,
        late_count=late_count,
        not_seen_count=not_seen_count,
        unknown_count=telemetry["unknown_count"],
        active_track_count=len(active_tracks_list),
        fps=telemetry["fps"],
        system_status=system_status,
        active_tracks=active_tracks_list
    )
    return ApiResponse.ok(summary)


# =============================================================================
# 6. Export APIs (CSV & JSON)
# =============================================================================

@router.get("/export/attendance/{session_id}/csv")
def export_attendance_csv(
    session_id: str = Path(...),
    state: AppState = Depends(get_app_state)
):
    """
    Exports session attendance in RFC 4180 CSV format.
    Fields: Session ID, Date, Class, Subject, Student ID, Register Number, Student Name, Status, First Seen, Last Seen.
    Guarantees: Zero embeddings, vectors, raw images, or biometric blobs.
    """
    sess = state.session_manager.get_session_info(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    records = state.db.get_attendance_for_session(session_id)

    output = io.StringIO()
    writer = csv.writer(output)

    # Standard academic header
    writer.writerow([
        "Session ID",
        "Date",
        "Class",
        "Subject",
        "Student ID",
        "Register Number",
        "Student Name",
        "Status",
        "First Seen",
        "Last Seen"
    ])

    for r in records:
        writer.writerow([
            sess.session_id,
            sess.date,
            sess.class_section,
            sess.subject,
            r.get("student_id", ""),
            r.get("student_id", ""),  # Register Number maps to student_id
            r.get("student_name", "") or "Unknown",
            r.get("status", ""),
            r.get("first_seen", ""),
            r.get("last_seen", "")
        ])

    csv_data = output.getvalue()
    filename = f"attendance_{sess.session_id}.csv"

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@router.get("/export/attendance/{session_id}/json", response_model=ApiResponse[AttendanceExportReportJson])
def export_attendance_json(
    session_id: str = Path(...),
    state: AppState = Depends(get_app_state)
):
    """
    Structured JSON attendance export.
    Guarantees: Zero biometric vectors, embeddings, or face image crops.
    """
    sess = state.session_manager.get_session_info(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    records = state.db.get_attendance_for_session(session_id)

    export_records: List[AttendanceExportRecordJson] = []
    for r in records:
        export_records.append(AttendanceExportRecordJson(
            student_id=r.get("student_id", ""),
            register_no=r.get("student_id", ""),
            name=r.get("student_name", "") or "Unknown",
            status=r.get("status", ""),
            first_seen=r.get("first_seen", ""),
            last_seen=r.get("last_seen", "")
        ))

    report = AttendanceExportReportJson(
        session=AttendanceExportSessionJson(
            session_id=sess.session_id,
            date=sess.date,
            class_section=sess.class_section,
            subject=sess.subject
        ),
        attendance=export_records
    )
    return ApiResponse.ok(report)


# =============================================================================
# 7. Real-Time Video & SSE Streaming
# =============================================================================

@router.get("/video/feed")
def get_video_feed(
    view_mode: str = Query("normal", pattern="^(normal|debug|raw|detector)$"),
    limit: Optional[int] = Query(None, ge=1, description="Optional frame limit for testing or single-shot"),
    state: AppState = Depends(get_app_state)
):
    """
    MJPEG live video stream:
    - view_mode=normal: Clean Smart Board cards (Student Name, Reg No, UNKNOWN).
    - view_mode=debug: Developer HUD (Track ID, bbox, similarity, quality, FPS, latency).
    - view_mode=raw: Pure YOLOv8 face detector boxes & keypoints before tracking/recognition.
    """
    def frame_generator():
        count = 0

        while True:
            if limit is not None and count >= limit:
                break
            frame_bytes = state.get_mjpeg_frame(view_mode=view_mode)
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
                count += 1
            time.sleep(0.04)  # ~25 FPS streaming

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/events/sse")
async def get_events_sse(
    limit: Optional[int] = Query(None, ge=1, description="Optional event limit for testing"),
    state: AppState = Depends(get_app_state)
):
    """
    Server-Sent Events (SSE) telemetry feed.
    Pushes live metric changes, counts, tracks, and session updates to dashboard.
    """
    async def event_generator():
        count = 0
        while True:
            if limit is not None and count >= limit:
                break
            telemetry = state.get_latest_telemetry()
            active_sess = state.session_manager.get_active_session()

            total_enrolled = state.db.get_student_count()
            present_count = 0
            late_count = 0
            not_seen_count = total_enrolled

            if active_sess:
                report = state.attendance_engine.generate_session_report(active_sess.session_id)
                present_count = report.present_count
                late_count = report.late_count
                not_seen_count = report.not_seen_count

            payload = {
                "session": active_sess.model_dump() if active_sess else None,
                "session_status": active_sess.status if active_sess else "NO_ACTIVE_SESSION",
                "total_enrolled": total_enrolled,
                "present_count": present_count,
                "late_count": late_count,
                "not_seen_count": not_seen_count,
                "unknown_count": telemetry["unknown_count"],
                "fps": round(telemetry["fps"], 1),
                "active_track_count": len(telemetry["tracks"]),
                "camera_online": telemetry["camera_online"],
                "timestamp": time.time()
            }

            yield f"data: {json.dumps(payload)}\n\n"
            count += 1
            if limit is not None and count >= limit:
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


# =============================================================================
# 8. Phase 10: Real Student Enrollment Endpoints
# =============================================================================

class FilePathRequest(BaseModel):
    file_path: str


@router.post("/enrollment/validate-file", response_model=ApiResponse[EnrollmentReportResponse])
def validate_enrollment_file(
    payload: FilePathRequest,
    state: AppState = Depends(get_app_state)
):
    """
    Dry-run validation on a student data file (DOCX, CSV, Directory).
    Performs face detection, single-face validation, and quality assessment without modifying the database.
    """
    if not os.path.exists(payload.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {payload.file_path}")

    try:
        importer = StudentDataImporter()
        records = importer.parse_file(payload.file_path)
    except ImporterValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    service = state.get_enrollment_service()
    report = service.enroll_batch(records, dry_run=True)
    return ApiResponse.ok(report)


@router.post("/enrollment/import", response_model=ApiResponse[EnrollmentReportResponse])
def import_and_enroll(
    payload: FilePathRequest,
    auth: AuthContext = Depends(require_operator),
    state: AppState = Depends(get_app_state)
):
    """
    Batch enrollment of student identities from an input document (DOCX, CSV, Directory).
    Operator authorization required.
    Validates single face, checks quality, computes ArcFace embeddings, and synchronizes SQLite & FAISS.
    """
    if not os.path.exists(payload.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {payload.file_path}")

    try:
        importer = StudentDataImporter()
        records = importer.parse_file(payload.file_path)
    except ImporterValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    service = state.get_enrollment_service()
    report = service.enroll_batch(records, dry_run=False)
    return ApiResponse.ok(report)


@router.post("/enrollment/student", response_model=ApiResponse[Dict[str, Any]])
def enroll_single_student(
    payload: EnrollStudentRequest,
    auth: AuthContext = Depends(require_operator),
    state: AppState = Depends(get_app_state)
):
    """
    Enrolls a single student with an optional base64-encoded image.
    Operator authorization required.
    """
    service = state.get_enrollment_service()
    service.register_student(
        student_id=payload.student_id,
        student_name=payload.student_name,
        department=payload.department,
        section=payload.section,
        reg_no=payload.register_no,
        class_section=payload.class_section
    )

    if payload.image_base64:
        try:
            img_bytes = base64.b64decode(payload.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data.")

        v_res = service.validation_pipeline.validate_photo(img_bytes, payload.student_id, payload.sample_label)
        if not v_res.is_valid:
            raise HTTPException(status_code=400, detail=f"Photo rejected: {v_res.rejection_reason}")

        success, emb_id, msg = service.enroll_face_sample(
            student_id=payload.student_id,
            face_tensor=v_res.aligned_tensor,
            sample_label=payload.sample_label,
            quality_score=v_res.quality_score
        )
        if not success:
            raise HTTPException(status_code=400, detail=msg)

        service.vector_store.save_index()
        return ApiResponse.ok({
            "student_id": payload.student_id,
            "status": "ENROLLED",
            "embedding_id": emb_id,
            "quality_score": v_res.quality_score
        })

    return ApiResponse.ok({
        "student_id": payload.student_id,
        "status": "REGISTERED_WITHOUT_PHOTO"
    })


@router.delete("/enrollment/student/{student_id}", response_model=ApiResponse[Dict[str, Any]])
def delete_student_endpoint(
    student_id: str = Path(...),
    auth: AuthContext = Depends(require_operator),
    state: AppState = Depends(get_app_state)
):
    """Removes a student and all associated embeddings from SQLite and FAISS."""
    service = state.get_enrollment_service()
    success = service.delete_student(student_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' not found.")
    return ApiResponse.ok({"student_id": student_id, "deleted": True})


@router.post("/enrollment/rebuild", response_model=ApiResponse[Dict[str, Any]])
def rebuild_faiss_index(
    auth: AuthContext = Depends(require_operator),
    state: AppState = Depends(get_app_state)
):
    """Rebuilds and re-synchronizes the FAISS index from the SQLite database."""
    service = state.get_enrollment_service()
    success, msg = service.rebuild_index()
    if not success:
        raise HTTPException(status_code=500, detail=msg)
    return ApiResponse.ok({"success": True, "message": msg, "total_vectors": service.vector_store.total_vectors})


@router.get("/enrollment/status", response_model=ApiResponse[Dict[str, Any]])
def get_enrollment_status(state: AppState = Depends(get_app_state)):
    """Returns enrollment database statistics and FAISS / SQLite consistency status."""
    service = state.get_enrollment_service()
    res = service.verify_index_consistency()
    total_students = service.db.get_student_count()
    total_embs = service.db.get_embedding_count()
    total_faiss = service.vector_store.total_vectors

    return ApiResponse.ok({
        "total_students": total_students,
        "total_embeddings_db": total_embs,
        "total_vectors_faiss": total_faiss,
        "consistent": res["consistent"],
        "is_consistent": res["consistent"],
        "consistency_message": res["message"]
    })


# =============================================================================
# =============================================================================
# 9. Camera Lifecycle & Source Control
# =============================================================================

@router.get("/camera/sources", response_model=ApiResponse[Dict[str, Any]])
def get_camera_sources(state: AppState = Depends(get_app_state)):
    """Returns available camera sources, active source, and DroidCam configuration."""
    settings = get_settings().camera
    active_source = "laptop"
    if state.camera_manager:
        active_source = getattr(state.camera_manager, "current_source", settings.source)
    else:
        active_source = settings.source

    sources = [
        {"id": "droidcam", "label": "DroidCam Wi-Fi", "type": "ip_stream", "description": "Mobile phone camera via Wi-Fi"},
        {"id": "laptop", "label": "Laptop Camera", "type": "uvc_index", "index": settings.laptop_index, "description": "Built-in laptop webcam"},
        {"id": "smart_board", "label": "Smart Board Camera", "type": "uvc_index", "index": settings.smart_board_index, "description": "Classroom Smart Board camera"},
        {"id": "external", "label": "External Camera", "type": "uvc_index", "index": settings.external_index, "description": "External USB/UVC camera"}
    ]

    dc = settings.droidcam
    return ApiResponse.ok({
        "active_source": active_source,
        "sources": sources,
        "droidcam_config": {
            "enabled": dc.enabled,
            "host": dc.host,
            "port": dc.port,
            "video_path": dc.video_path,
            "base_url": dc.base_url,
            "video_url": dc.video_url
        }
    })


@router.post("/camera/select", response_model=ApiResponse[Dict[str, Any]])
def select_camera_source(
    payload: SelectCameraRequest,
    state: AppState = Depends(get_app_state)
):
    """
    Dynamically switches active camera source (e.g. DroidCam Wi-Fi, Laptop Camera).
    Does NOT restart FastAPI or disrupt attendance database state.
    """
    kwargs = {}
    if payload.host:
        kwargs["host"] = payload.host
    if payload.port:
        kwargs["port"] = payload.port
    if payload.video_path:
        kwargs["video_path"] = payload.video_path
    if payload.index is not None:
        kwargs["index"] = payload.index

    success, message = state.switch_camera_source(payload.source, **kwargs)
    status_info = state.camera_manager.get_status() if state.camera_manager else {}

    return ApiResponse.ok({
        "source": payload.source,
        "success": success,
        "message": message,
        "camera_online": state.camera_online,
        "details": status_info
    })


@router.post("/camera/test", response_model=ApiResponse[Dict[str, Any]])
def test_camera_connection(
    payload: TestCameraRequest,
    state: AppState = Depends(get_app_state)
):
    """
    Tests connectivity to specified camera or DroidCam IP without interrupting active session.
    Probes TCP port and verifies stream accessibility.
    """
    from camera.droidcam_camera import DroidCamCamera

    host = payload.host or "10.140.159.218"
    port = payload.port or 4747
    video_path = payload.video_path or "/video"

    reachable, msg = DroidCamCamera.probe_endpoint(host, port, timeout=1.5)
    if not reachable:
        return ApiResponse.ok({
            "connected": False,
            "status": "NOT CONNECTED",
            "host": host,
            "port": port,
            "video_url": f"http://{host}:{port}{video_path}",
            "message": f"DroidCam video stream unavailable at http://{host}:{port}{video_path}. ({msg})",
            "reasons": [
                "Phone and laptop not connected to the same Wi-Fi network",
                "DroidCam mobile app is not currently open/streaming on the phone",
                f"Incorrect phone Wi-Fi IP address ({host})",
                "Windows Firewall is blocking incoming connections on port 4747"
            ]
        })

    # If TCP port is reachable, perform short test capture
    test_cam = DroidCamCamera(host=host, port=port, video_path=video_path)
    is_conn = test_cam.is_connected
    w, h = test_cam.resolution
    fps = test_cam.reported_fps
    test_cam.release()

    if is_conn:
        return ApiResponse.ok({
            "connected": True,
            "status": "CONNECTED",
            "host": host,
            "port": port,
            "video_url": f"http://{host}:{port}{video_path}",
            "resolution": f"{w}x{h}",
            "fps": fps,
            "message": f"Successfully connected to DroidCam video stream ({w}x{h} @ {fps:.1f} FPS)."
        })
    else:
        return ApiResponse.ok({
            "connected": False,
            "status": "NOT CONNECTED",
            "host": host,
            "port": port,
            "video_url": f"http://{host}:{port}{video_path}",
            "message": "TCP port reached, but video frame decoding failed. Verify that video streaming is active in DroidCam."
        })


@router.get("/camera/status", response_model=ApiResponse[Dict[str, Any]])
def get_camera_status(state: AppState = Depends(get_app_state)):
    """Returns real-time status of physical camera hardware, active source, and worker thread."""
    cam_status = {}
    active_source = "unknown"
    if state.camera_manager:
        try:
            cam_status = state.camera_manager.get_status()
            active_source = state.camera_manager.current_source
        except Exception:
            pass

    return ApiResponse.ok({
        "camera_online": state.camera_online,
        "worker_running": state._camera_thread is not None and state._camera_thread.is_alive(),
        "active_source": active_source,
        "fps": round(state.latest_fps, 1),
        "details": cam_status
    })


@router.post("/camera/start", response_model=ApiResponse[Dict[str, Any]])
def start_camera_feed(state: AppState = Depends(get_app_state)):
    """Starts or restarts the background camera capture and face identification pipeline."""
    state.start_camera_worker()
    return ApiResponse.ok({
        "message": "Camera background worker started",
        "camera_online": state.camera_online
    })


@router.post("/camera/stop", response_model=ApiResponse[Dict[str, Any]])
def stop_camera_feed(state: AppState = Depends(get_app_state)):
    """Stops the camera capture worker and releases physical camera hardware."""
    state.stop_camera_worker()
    return ApiResponse.ok({
        "message": "Camera worker stopped and hardware released",
        "camera_online": False
    })




