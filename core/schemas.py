from typing import List, Optional, Any
from pydantic import BaseModel, Field

class FaceQualityResult(BaseModel):
    """
    Standardized result for face quality assessment.
    """
    face_id: str = Field(description="Unique identifier for this detection instance")
    bbox: List[int] = Field(description="[x1, y1, x2, y2] bounding box coordinates")
    keypoints: Optional[List[List[float]]] = Field(default=None, description="5 facial landmarks: [[x,y], [x,y], ...]")
    width: int
    height: int
    area: int
    sharpness: float = Field(description="Variance of Laplacian score")
    brightness: float = Field(description="Mean pixel intensity")
    pose: Optional[str] = Field(default="UNKNOWN", description="Pose estimation if available")
    quality_status: str = Field(description="RECOGNITION_READY or LOW_QUALITY")
    rejection_reason: Optional[str] = Field(default=None, description="Reason for rejection if LOW_QUALITY")

class RecognitionReadyFace(BaseModel):
    """
    The final output of Phase 5, ready to be passed to the Phase 6 Recognizer.
    """
    quality_metrics: FaceQualityResult
    original_bbox: List[int]
    aligned_face_tensor: Any = Field(description="The 112x112 normalized numpy array ready for ArcFace")
    timestamp: float = Field(description="Frame timestamp")

class RecognitionStatus:
    MATCH = "MATCH"
    UNKNOWN = "UNKNOWN"
    INVALID = "INVALID"
    ERROR = "ERROR"

class RecognitionResult(BaseModel):
    """
    Standardized output of Phase 6 Face Recognizer.
    """
    face_id: str = Field(description="Unique face ID from detector/tracker")
    matched_student_id: Optional[str] = Field(default=None, description="Enrolled Student ID if MATCH, None if UNKNOWN")
    matched_student_name: Optional[str] = Field(default=None, description="Enrolled Student Name if MATCH")
    similarity: float = Field(description="Cosine similarity score [-1.0, 1.0]")
    status: str = Field(description="MATCH, UNKNOWN, INVALID, or ERROR")
    threshold: float = Field(description="Threshold used for matching decision")
    embedding_model: str = Field(description="ArcFace model identifier")
    processing_time_ms: float = Field(description="Inference + search latency in milliseconds")
    bbox: List[int] = Field(default_factory=list, description="Original bounding box")
    quality_metrics: Optional[FaceQualityResult] = Field(default=None, description="Phase 5 quality metrics")

class EnrollmentSample(BaseModel):
    """
    Single face sample used during enrollment.
    """
    sample_label: str = Field(default="frontal", description="Description e.g. frontal, left, right")
    aligned_face_tensor: Any = Field(description="112x112 normalized tensor")
    quality_score: float = Field(default=1.0, description="Quality metric score")

class StudentProfile(BaseModel):
    """
    Enrolled student profile metadata.
    """
    student_id: str
    student_name: str
    department: str = "Computer Science"
    section: str = "A"
    status: str = "active"
    sample_count: int = 0

class TrackState:
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    LOST = "LOST"
    REMOVED = "REMOVED"

class RecognitionObservation(BaseModel):
    """
    Single observation of a recognition result for a tracked face.
    """
    frame_id: int
    timestamp: float
    status: str
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    similarity: float = 0.0
    quality_status: str = "RECOGNITION_READY"
    sharpness: float = 0.0

class TrackedFace(BaseModel):
    """
    Phase 7 Output: Unified tracked entity with temporal identity stabilization.
    """
    track_id: int
    bbox: List[int] = Field(description="[x1, y1, x2, y2] bounding box")
    state: str = Field(default=TrackState.NEW, description="NEW, ACTIVE, LOST, REMOVED")
    score: float = Field(default=0.0, description="Detection confidence")
    stable_student_id: Optional[str] = Field(default=None, description="Stabilized Student ID (None if UNKNOWN)")
    stable_student_name: Optional[str] = Field(default=None, description="Stabilized Student Name")
    current_status: str = Field(default=RecognitionStatus.UNKNOWN, description="Latest raw recognition status")
    current_similarity: float = Field(default=0.0, description="Latest similarity score")
    quality_status: str = Field(default="RECOGNITION_READY", description="Latest quality status")
    hits: int = Field(default=1, description="Number of successful matches")
    age: int = Field(default=1, description="Total frames since track inception")
    time_since_update: int = Field(default=0, description="Frames since last detection match")
    last_seen: float = Field(default=0.0, description="Timestamp of last detection")
    history_len: int = Field(default=0, description="Observations recorded in temporal buffer")
    quality_metrics: Optional[FaceQualityResult] = Field(default=None, description="Latest quality assessment")

# ===========================================================================
# Phase 8 Schemas: Session Management & Attendance Engine
# ===========================================================================

class SessionState:
    SCHEDULED = "SCHEDULED"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    ENDED = "ENDED"
    CANCELLED = "CANCELLED"
    RECOVERING = "RECOVERING"

class AttendanceStatus:
    PRESENT = "PRESENT"
    LATE = "LATE"
    NOT_SEEN = "NOT_SEEN"
    UNKNOWN = "UNKNOWN"
    INVALID = "INVALID"

class SessionInfo(BaseModel):
    """
    Represents an instructional class session.
    """
    session_id: str
    date: str = Field(description="Date formatted YYYY-MM-DD")
    class_section: str = Field(description="e.g. AIDS-B")
    subject: str = Field(description="e.g. Deep Learning")
    planned_start_time: str = Field(description="ISO or HH:MM string")
    planned_end_time: str = Field(description="ISO or HH:MM string")
    actual_start_time: Optional[str] = None
    actual_end_time: Optional[str] = None
    status: str = Field(default=SessionState.SCHEDULED)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class AttendanceRecord(BaseModel):
    """
    Normalized attendance record for one student in one session.
    """
    attendance_id: Optional[int] = None
    session_id: str
    student_id: str
    student_name: Optional[str] = None
    status: str = Field(description="PRESENT or LATE")
    first_seen: str = Field(description="ISO timestamp of first valid recognition")
    last_seen: str = Field(description="ISO timestamp of latest valid recognition")
    first_track_id: Optional[int] = None
    last_track_id: Optional[int] = None
    initial_similarity: float = 0.0
    latest_similarity: float = 0.0
    marked_at: Optional[str] = None
    updated_at: Optional[str] = None

class AttendanceEvent(BaseModel):
    """
    Decoupled attendance event generated by the Phase 7 recognition/tracking pipeline.
    """
    session_id: str
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    track_id: int
    timestamp: float = Field(description="Epoch timestamp in seconds")
    recognition_status: str = Field(description="MATCH, UNKNOWN, INVALID, ERROR")
    similarity: float = 0.0
    quality_status: str = "RECOGNITION_READY"
    source: str = "tracking_pipeline"

class SessionAttendanceReport(BaseModel):
    """
    Full session attendance summary and roster reconciliation.
    """
    session: SessionInfo
    total_enrolled: int
    present_count: int
    late_count: int
    not_seen_count: int
    records: List[AttendanceRecord] = Field(default_factory=list)
    not_seen_students: List[str] = Field(default_factory=list)



