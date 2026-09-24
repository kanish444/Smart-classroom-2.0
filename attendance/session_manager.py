import datetime
from typing import Dict, Any, Optional, List
from loguru import logger

from database.db_manager import DatabaseManager
from core.schemas import SessionState, SessionInfo
from config.settings import get_settings


class SessionManagerError(Exception):
    """Custom exception for invalid session operations."""
    pass


class InvalidSessionTransitionError(SessionManagerError):
    """Raised when an illegal lifecycle transition is attempted."""
    pass


class DuplicateSessionError(SessionManagerError):
    """Raised when a session ID already exists or overlapping active sessions collide."""
    pass


class SessionManager:
    """
    Manages the lifecycle and state machine of instructional class sessions:
    SCHEDULED -> ACTIVE -> (PAUSED) -> ENDED (or CANCELLED)
    Enforces strict state validation, prevents overlapping sessions, and handles crash recovery.
    """

    # Strict transition table: current_state -> set of allowed next_states
    ALLOWED_TRANSITIONS = {
        SessionState.SCHEDULED: {SessionState.ACTIVE, SessionState.CANCELLED},
        SessionState.ACTIVE: {SessionState.PAUSED, SessionState.ENDED, SessionState.CANCELLED},
        SessionState.PAUSED: {SessionState.ACTIVE, SessionState.ENDED, SessionState.CANCELLED},
        SessionState.RECOVERING: {SessionState.ACTIVE, SessionState.ENDED, SessionState.CANCELLED},
        SessionState.ENDED: set(),       # Terminal state
        SessionState.CANCELLED: set()   # Terminal state
    }

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.settings = get_settings().session
        self.db = db_manager or DatabaseManager()
        self.timezone_str = self.settings.timezone

    def _now_iso(self) -> str:
        """Returns current local timestamp in ISO format."""
        return datetime.datetime.now().isoformat()

    def create_session(
        self,
        session_id: str,
        date: str,
        class_section: str,
        subject: str,
        planned_start_time: str,
        planned_end_time: str
    ) -> SessionInfo:
        """
        Creates a new instructional session in SCHEDULED state.
        Validates date format and uniqueness.
        """
        if not session_id or not session_id.strip():
            raise SessionManagerError("Session ID cannot be empty.")

        # Check existing session
        existing = self.db.get_session(session_id)
        if existing is not None:
            raise DuplicateSessionError(f"Session with ID '{session_id}' already exists.")

        # Validate date format YYYY-MM-DD
        try:
            datetime.date.fromisoformat(date)
        except ValueError:
            raise SessionManagerError(f"Invalid date format '{date}'. Expected YYYY-MM-DD.")

        success = self.db.create_session(
            session_id=session_id,
            date=date,
            class_section=class_section,
            subject=subject,
            planned_start_time=planned_start_time,
            planned_end_time=planned_end_time,
            status=SessionState.SCHEDULED
        )
        if not success:
            raise SessionManagerError(f"Database failed to create session '{session_id}'.")

        logger.info(f"SESSION_CREATED: {session_id} [{class_section} - {subject}] on {date}")
        sess_dict = self.db.get_session(session_id)
        return SessionInfo(**sess_dict)

    def _validate_transition(self, current_status: str, target_status: str):
        allowed = self.ALLOWED_TRANSITIONS.get(current_status, set())
        if target_status not in allowed:
            raise InvalidSessionTransitionError(
                f"Invalid session state transition from '{current_status}' to '{target_status}'."
            )

    def start_session(self, session_id: str, actual_start_time: Optional[str] = None) -> SessionInfo:
        """
        Transitions session to ACTIVE.
        Ensures no other session for the same class_section is currently ACTIVE.
        """
        session = self.db.get_session(session_id)
        if not session:
            raise SessionManagerError(f"Session '{session_id}' not found.")

        current_status = session["status"]
        self._validate_transition(current_status, SessionState.ACTIVE)

        # Check collision: no other active session for the same class_section
        active_sess = self.db.get_active_session(session["class_section"])
        if active_sess and active_sess["session_id"] != session_id:
            raise DuplicateSessionError(
                f"Class/section '{session['class_section']}' already has an ACTIVE session: '{active_sess['session_id']}'."
            )

        start_time = actual_start_time or self._now_iso()
        self.db.update_session_status(
            session_id=session_id,
            status=SessionState.ACTIVE,
            actual_start_time=start_time
        )
        logger.info(f"SESSION_STARTED: {session_id} at {start_time}")
        return SessionInfo(**self.db.get_session(session_id))

    def pause_session(self, session_id: str) -> SessionInfo:
        """Transitions an ACTIVE session to PAUSED."""
        session = self.db.get_session(session_id)
        if not session:
            raise SessionManagerError(f"Session '{session_id}' not found.")

        self._validate_transition(session["status"], SessionState.PAUSED)
        self.db.update_session_status(session_id=session_id, status=SessionState.PAUSED)
        logger.info(f"SESSION_PAUSED: {session_id}")
        return SessionInfo(**self.db.get_session(session_id))

    def resume_session(self, session_id: str) -> SessionInfo:
        """Resumes a PAUSED or RECOVERING session back to ACTIVE."""
        session = self.db.get_session(session_id)
        if not session:
            raise SessionManagerError(f"Session '{session_id}' not found.")

        self._validate_transition(session["status"], SessionState.ACTIVE)
        self.db.update_session_status(session_id=session_id, status=SessionState.ACTIVE)
        logger.info(f"SESSION_RESUMED: {session_id}")
        return SessionInfo(**self.db.get_session(session_id))

    def end_session(self, session_id: str, actual_end_time: Optional[str] = None) -> SessionInfo:
        """Transitions an ACTIVE or PAUSED session to ENDED (terminal state)."""
        session = self.db.get_session(session_id)
        if not session:
            raise SessionManagerError(f"Session '{session_id}' not found.")

        self._validate_transition(session["status"], SessionState.ENDED)
        end_time = actual_end_time or self._now_iso()
        self.db.update_session_status(
            session_id=session_id,
            status=SessionState.ENDED,
            actual_end_time=end_time
        )
        logger.info(f"SESSION_ENDED: {session_id} at {end_time}")
        return SessionInfo(**self.db.get_session(session_id))

    def cancel_session(self, session_id: str) -> SessionInfo:
        """Cancels a session (terminal state)."""
        session = self.db.get_session(session_id)
        if not session:
            raise SessionManagerError(f"Session '{session_id}' not found.")

        self._validate_transition(session["status"], SessionState.CANCELLED)
        self.db.update_session_status(session_id=session_id, status=SessionState.CANCELLED)
        logger.info(f"SESSION_CANCELLED: {session_id}")
        return SessionInfo(**self.db.get_session(session_id))

    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Retrieves strongly-typed SessionInfo by ID."""
        row = self.db.get_session(session_id)
        return SessionInfo(**row) if row else None

    def get_active_session(self, class_section: Optional[str] = None) -> Optional[SessionInfo]:
        """Retrieves current active session, if any."""
        row = self.db.get_active_session(class_section=class_section)
        return SessionInfo(**row) if row else None

    def recover_sessions(self, reference_time: Optional[datetime.datetime] = None) -> List[SessionInfo]:
        """
        Scans for unclosed sessions (ACTIVE, PAUSED, RECOVERING) upon application startup.
        - If current time exceeds planned_end_time: marks ENDED.
        - Else: marks RECOVERING to allow graceful resume.
        """
        now = reference_time or datetime.datetime.now()
        unclosed = self.db.get_unclosed_sessions()
        recovered = []

        for sess in unclosed:
            sid = sess["session_id"]
            # Attempt to parse planned end datetime
            date_str = sess["date"]
            end_time_str = sess["planned_end_time"]

            is_expired = False
            try:
                # Support both full ISO and HH:MM format
                if "T" in end_time_str:
                    end_dt = datetime.datetime.fromisoformat(end_time_str)
                else:
                    parts = end_time_str.split(":")
                    end_dt = datetime.datetime.fromisoformat(date_str).replace(
                        hour=int(parts[0]), minute=int(parts[1]), second=int(parts[2]) if len(parts) > 2 else 0
                    )
                if now > end_dt:
                    is_expired = True
            except Exception as e:
                logger.debug(f"Could not parse planned end time '{end_time_str}': {e}")

            if is_expired:
                self.db.update_session_status(
                    session_id=sid,
                    status=SessionState.ENDED,
                    actual_end_time=now.isoformat()
                )
                logger.info(f"SESSION_RECOVERY: Auto-closed expired unclosed session '{sid}' to ENDED.")
            else:
                self.db.update_session_status(session_id=sid, status=SessionState.RECOVERING)
                logger.info(f"SESSION_RECOVERY: Marked unclosed session '{sid}' as RECOVERING.")

            recovered.append(SessionInfo(**self.db.get_session(sid)))

        return recovered
