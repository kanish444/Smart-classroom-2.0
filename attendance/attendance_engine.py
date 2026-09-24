import time
import datetime
from typing import Dict, Any, Optional, List, Tuple, Set
from loguru import logger

from database.db_manager import DatabaseManager
from attendance.session_manager import SessionManager
from core.schemas import (
    TrackedFace,
    RecognitionStatus,
    AttendanceRecord,
    AttendanceEvent,
    SessionAttendanceReport,
    SessionState,
    AttendanceStatus,
    SessionInfo
)
from config.settings import get_settings


class AttendanceEngineError(Exception):
    """Base exception for attendance engine operations."""
    pass


class AttendanceEngine:
    """
    Phase 8 Attendance Engine:
    - Consumes stabilized recognition outputs from Phase 7 (TrackedFace or AttendanceEvent)
    - Enforces strict identity decoupling: Track ID != Student ID
    - Calculates PRESENT vs LATE based on configurable late threshold policies
    - Manages pre-session and post-session allowable time windows
    - Preserves first_seen and updates last_seen across multi-frame recognitions
    - In-memory event deduplication suppresses redundant sub-second database transactions
    - Strictly rejects UNKNOWN, INVALID, and ERROR entities
    - Reconciles roster to produce complete SessionAttendanceReports with NOT_SEEN tracking
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        session_manager: Optional[SessionManager] = None
    ):
        self.settings = get_settings().attendance
        self.session_settings = get_settings().session
        self.db = db_manager or DatabaseManager()
        self.session_manager = session_manager or SessionManager(db_manager=self.db)

        self.late_threshold_minutes = self.settings.late_threshold_minutes
        self.allow_pre_session_marking = self.settings.allow_pre_session_marking
        self.min_similarity = self.settings.min_similarity_threshold
        self.dedup_buffer_seconds = self.settings.deduplication_buffer_seconds

        # In-memory deduplication cache: (session_id, student_id) -> last_processed_timestamp
        self._dedup_cache: Dict[Tuple[str, str], float] = {}

    def clear_cache(self):
        """Clears in-memory deduplication cache."""
        self._dedup_cache.clear()

    def _parse_time(self, date_str: str, time_str: str) -> datetime.datetime:
        """Helper to parse a date and time string into a datetime object."""
        if "T" in time_str:
            return datetime.datetime.fromisoformat(time_str)
        parts = time_str.split(":")
        return datetime.datetime.fromisoformat(date_str).replace(
            hour=int(parts[0]), minute=int(parts[1]), second=int(parts[2]) if len(parts) > 2 else 0
        )

    def process_tracked_faces(
        self,
        session_id: str,
        tracked_faces: List[TrackedFace],
        timestamp: Optional[float] = None
    ) -> List[AttendanceRecord]:
        """
        Consumes a frame's worth of TrackedFace outputs from the Phase 7 TrackingPipeline.
        Filters out UNKNOWN, unconfirmed, or invalid faces and records valid attendances.
        """
        records: List[AttendanceRecord] = []
        now_ts = timestamp or time.time()

        for tf in tracked_faces:
            # 1. Strictly ignore UNKNOWN, UNCONFIRMED, or INVALID tracks
            if not tf.stable_student_id or tf.stable_student_id.strip() == "":
                logger.debug(f"UNKNOWN_IGNORED: Track {tf.track_id} has no confirmed student ID.")
                continue

            if tf.current_status in (RecognitionStatus.INVALID, RecognitionStatus.ERROR):
                logger.debug(f"INVALID_IGNORED: Track {tf.track_id} reported status {tf.current_status}.")
                continue

            # 2. Dispatch decoupled attendance event
            event = AttendanceEvent(
                session_id=session_id,
                student_id=tf.stable_student_id,
                student_name=tf.stable_student_name,
                track_id=tf.track_id,
                timestamp=now_ts,
                recognition_status=tf.current_status,
                similarity=tf.current_similarity,
                quality_status=tf.quality_status,
                source="tracking_pipeline"
            )

            record = self.process_attendance_event(event)
            if record is not None:
                records.append(record)

        return records

    def process_attendance_event(self, event: AttendanceEvent) -> Optional[AttendanceRecord]:
        """
        Processes a single AttendanceEvent:
        - Validates session state & allowed windows
        - Deduplicates sub-second redundant events
        - Evaluates PRESENT vs LATE
        - Atomically persists or updates SQLite record
        """
        # Validate student existence in database
        student = self.db.get_student(event.student_id)
        if not student:
            logger.warning(f"Attendance rejected: Student '{event.student_id}' does not exist in student database.")
            return None

        # Validate similarity threshold
        if event.similarity < self.min_similarity:
            logger.debug(f"Attendance rejected: Similarity {event.similarity:.3f} below threshold {self.min_similarity}.")
            return None

        # Retrieve and validate session
        session = self.session_manager.get_session_info(event.session_id)
        if not session:
            logger.error(f"Attendance rejected: Session '{event.session_id}' not found.")
            return None

        # Check session status
        if session.status in (SessionState.ENDED, SessionState.CANCELLED):
            logger.warning(f"Attendance rejected: Session '{event.session_id}' is in terminal state '{session.status}'.")
            return None

        event_dt = datetime.datetime.fromtimestamp(event.timestamp)
        planned_start_dt = self._parse_time(session.date, session.planned_start_time)
        planned_end_dt = self._parse_time(session.date, session.planned_end_time)

        # Check pre-session window if session is still SCHEDULED
        if session.status == SessionState.SCHEDULED:
            pre_window = datetime.timedelta(minutes=self.session_settings.pre_session_window_minutes)
            if self.allow_pre_session_marking and (planned_start_dt - pre_window <= event_dt <= planned_end_dt):
                pass  # Allowed during pre-session window
            else:
                logger.debug(f"Attendance rejected: Session '{event.session_id}' is SCHEDULED and outside pre-session window.")
                return None

        # Sub-second Deduplication Check
        cache_key = (event.session_id, event.student_id)
        last_time = self._dedup_cache.get(cache_key)
        if last_time is not None and (event.timestamp - last_time < self.dedup_buffer_seconds):
            # Rapid frame deduplication: return existing cached record without disk write
            raw_rec = self.db.get_attendance_record(event.session_id, event.student_id)
            return AttendanceRecord(**raw_rec) if raw_rec else None

        self._dedup_cache[cache_key] = event.timestamp

        # Determine PRESENT vs LATE status
        # Reference start time is actual_start_time if started, else planned_start_time
        ref_start_str = session.actual_start_time or session.planned_start_time
        ref_start_dt = self._parse_time(session.date, ref_start_str)

        late_threshold = ref_start_dt + datetime.timedelta(minutes=self.late_threshold_minutes)
        if event_dt <= late_threshold:
            status = AttendanceStatus.PRESENT
        else:
            status = AttendanceStatus.LATE

        event_iso = event_dt.isoformat()

        # Atomic Insert or Update
        success, is_new = self.db.record_or_update_attendance(
            session_id=event.session_id,
            student_id=event.student_id,
            status=status,
            first_seen=event_iso,
            last_seen=event_iso,
            track_id=event.track_id,
            similarity=event.similarity
        )

        if not success:
            logger.error(f"DATABASE_ERROR: Failed to record attendance for {event.student_id} in {event.session_id}.")
            return None

        if is_new:
            logger.info(f"ATTENDANCE_MARKED: Student {event.student_id} marked {status} in Session {event.session_id} (Track: {event.track_id}, Sim: {event.similarity:.2f}).")
        else:
            logger.debug(f"ATTENDANCE_UPDATED: Student {event.student_id} updated last_seen={event_iso} in Session {event.session_id}.")

        row = self.db.get_attendance_record(event.session_id, event.student_id)
        return AttendanceRecord(**row) if row else None

    def generate_session_report(self, session_id: str) -> SessionAttendanceReport:
        """
        Reconciles attendance records against the enrolled student roster for the session's class/section.
        Categorizes students into PRESENT, LATE, or NOT_SEEN.
        """
        session_info = self.session_manager.get_session_info(session_id)
        if not session_info:
            raise AttendanceEngineError(f"Session '{session_id}' not found.")

        # Get all enrolled students in this section
        all_students = self.db.get_all_students()
        enrolled_in_section = [
            s for s in all_students
            if s.get("section") == session_info.class_section or s.get("department") == session_info.class_section
            or session_info.class_section in (s.get("section", ""), s.get("department", ""), "ALL")
        ]
        # If class_section didn't filter directly, fall back to all enrolled students
        if not enrolled_in_section:
            enrolled_in_section = all_students

        enrolled_ids = {s["student_id"] for s in enrolled_in_section}

        # Retrieve recorded attendances
        raw_records = self.db.get_attendance_for_session(session_id)
        records = [AttendanceRecord(**r) for r in raw_records]

        seen_ids = {r.student_id for r in records}
        present_count = sum(1 for r in records if r.status == AttendanceStatus.PRESENT)
        late_count = sum(1 for r in records if r.status == AttendanceStatus.LATE)

        not_seen_ids = sorted(list(enrolled_ids - seen_ids))

        return SessionAttendanceReport(
            session=session_info,
            total_enrolled=len(enrolled_ids),
            present_count=present_count,
            late_count=late_count,
            not_seen_count=len(not_seen_ids),
            records=records,
            not_seen_students=not_seen_ids
        )
