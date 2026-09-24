import datetime
from typing import TypeVar, Generic, Optional, Any, Dict, List
from pydantic import BaseModel, Field

from core.schemas import (
    SessionInfo,
    AttendanceRecord,
    TrackedFace,
    SessionAttendanceReport
)

T = TypeVar("T")


class ErrorDetail(BaseModel):
    code: str
    message: str


class ApiResponse(BaseModel, Generic[T]):
    success: bool = True
    data: Optional[T] = None
    error: Optional[ErrorDetail] = None
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

    @classmethod
    def ok(cls, data: Any = None) -> "ApiResponse":
        return cls(success=True, data=data, error=None)

    @classmethod
    def fail(cls, code: str, message: str) -> "ApiResponse":
        return cls(success=False, data=None, error=ErrorDetail(code=code, message=message))


class PaginationMeta(BaseModel):
    page: int
    page_size: int
    total_items: int
    total_pages: int


class PaginatedData(BaseModel, Generic[T]):
    items: List[T]
    pagination: PaginationMeta


class CreateSessionRequest(BaseModel):
    session_id: str = Field(..., description="Unique session ID, e.g. SESS_20260923_AIDS_B_1")
    date: str = Field(..., description="Session date formatted YYYY-MM-DD")
    class_section: str = Field(..., description="Class & section, e.g. AIDS - B")
    subject: str = Field(..., description="Course subject, e.g. Python")
    planned_start_time: str = Field(..., description="Planned start time (ISO or HH:MM)")
    planned_end_time: str = Field(..., description="Planned end time (ISO or HH:MM)")


class StartSessionRequest(BaseModel):
    actual_start_time: Optional[str] = None


class EndSessionRequest(BaseModel):
    actual_end_time: Optional[str] = None
    confirm: bool = Field(default=True, description="Explicit operator confirmation to end session")


class AttendanceExportSessionJson(BaseModel):
    session_id: str
    date: str
    class_section: str
    subject: str


class AttendanceExportRecordJson(BaseModel):
    student_id: str
    register_no: str
    name: str
    status: str
    first_seen: str
    last_seen: str


class AttendanceExportReportJson(BaseModel):
    session: AttendanceExportSessionJson
    attendance: List[AttendanceExportRecordJson]



class HealthComponentStatus(BaseModel):
    status: str = "ONLINE"  # ONLINE, DEGRADED, OFFLINE
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class SystemHealthData(BaseModel):
    status: str = "ONLINE"
    components: Dict[str, HealthComponentStatus] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())


class ActiveTrackItem(BaseModel):
    track_id: int
    bbox: List[int]
    state: str
    stable_student_id: Optional[str] = None
    stable_student_name: Optional[str] = None
    current_similarity: float = 0.0
    quality_status: str = "RECOGNITION_READY"
    current_status: str = "UNKNOWN"
    hits: int = 1
    age: int = 1


class DashboardSummaryData(BaseModel):
    session: Optional[SessionInfo] = None
    session_status: str = "NO_ACTIVE_SESSION"
    total_enrolled: int = 0
    present_count: int = 0
    late_count: int = 0
    not_seen_count: int = 0
    unknown_count: int = 0
    active_track_count: int = 0
    fps: float = 0.0
    system_status: str = "ONLINE"
    active_tracks: List[ActiveTrackItem] = Field(default_factory=list)


class StudentProfileData(BaseModel):
    student_id: str
    student_name: str
    department: str = "Computer Science"
    section: str = "A"
    status: str = "active"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class StudentDetailData(BaseModel):
    profile: StudentProfileData
    attendance_history: List[Dict[str, Any]] = Field(default_factory=list)
    total_sessions_attended: int = 0


class RecognitionStatusData(BaseModel):
    model_name: str
    embedding_dim: int
    similarity_threshold: float
    device: str
    faiss_index_path: str
    index_total_vectors: int
    status: str = "ONLINE"


class StudentEnrollmentItemSchema(BaseModel):
    student_id: str
    name: str
    class_section: str
    status: str
    rejection_reason: Optional[str] = None
    samples_enrolled: int = 0
    embedding_ids: List[int] = Field(default_factory=list)


class EnrollmentReportResponse(BaseModel):
    total_records: int
    successfully_enrolled: int
    failed: int
    requires_review: int
    counts: Dict[str, int]
    details: List[StudentEnrollmentItemSchema] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    avg_time_per_student_ms: float = 0.0
    timestamp: str


class EnrollStudentRequest(BaseModel):
    student_id: str
    student_name: str
    register_no: Optional[str] = None
    class_section: Optional[str] = None
    department: str = "Computer Science"
    section: str = "A"
    sample_label: str = "frontal"
    image_base64: Optional[str] = None


class SelectCameraRequest(BaseModel):
    source: str = "droidcam"
    host: Optional[str] = None
    port: Optional[int] = None
    video_path: Optional[str] = None
    index: Optional[int] = None


class TestCameraRequest(BaseModel):
    source: str = "droidcam"
    host: Optional[str] = "10.140.159.218"
    port: Optional[int] = 4747
    video_path: Optional[str] = "/video"


