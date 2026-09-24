import os
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from loguru import logger
from config.settings import get_settings


class DatabaseManager:
    """
    Manages SQLite database storage for SmartClass Vision AI:
    - Student profiles and registration metadata
    - Face embeddings (serialized float32 binary BLOBs)
    - Model and threshold configuration metadata
    """

    def __init__(self, db_path: Optional[str] = None):
        self.settings = get_settings()
        self.db_path = db_path or self.settings.db_path

        # Ensure directory exists
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Returns an active SQLite connection with row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        """Initializes tables for students, embeddings, and model metadata."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    student_name TEXT NOT NULL,
                    department TEXT NOT NULL,
                    section TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    vector_blob BLOB NOT NULL,
                    quality_score REAL DEFAULT 1.0,
                    sample_label TEXT DEFAULT 'frontal',
                    model_version TEXT DEFAULT 'w600k_mbf',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    metric TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    class_section TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    planned_start_time TEXT NOT NULL,
                    planned_end_time TEXT NOT NULL,
                    actual_start_time TEXT,
                    actual_end_time TEXT,
                    status TEXT NOT NULL DEFAULT 'SCHEDULED',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions(date);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    first_track_id INTEGER,
                    last_track_id INTEGER,
                    initial_similarity REAL,
                    latest_similarity REAL,
                    marked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
                    UNIQUE (session_id, student_id)
                );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_session ON attendance(session_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_session_student ON attendance(session_id, student_id);")

            # Phase 10 schema extensions
            try:
                cursor.execute("ALTER TABLE students ADD COLUMN register_no TEXT;")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE students ADD COLUMN class_name TEXT;")
            except Exception:
                pass

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS enrollment_sources (
                    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    filename TEXT,
                    face_count INTEGER DEFAULT 1,
                    quality_score REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS enrollment_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    student_id TEXT,
                    details TEXT,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            conn.commit()

    def add_student(
        self,
        student_id: str,
        student_name: str,
        department: str = "Computer Science",
        section: str = "A",
        status: str = "active",
        register_no: Optional[str] = None,
        class_name: Optional[str] = None
    ) -> bool:
        """Enrolls or updates a student profile."""
        reg_no = register_no or student_id
        cls_name = class_name or f"{department} - {section}"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO students (student_id, student_name, department, section, status, register_no, class_name, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(student_id) DO UPDATE SET
                    student_name = excluded.student_name,
                    department = excluded.department,
                    section = excluded.section,
                    status = excluded.status,
                    register_no = excluded.register_no,
                    class_name = excluded.class_name,
                    updated_at = CURRENT_TIMESTAMP;
            """, (student_id, student_name, department, section, status, reg_no, cls_name))
            conn.commit()
            return True

    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single student profile by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM students WHERE student_id = ?;", (student_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            if not data.get("register_no"):
                data["register_no"] = data["student_id"]
            if not data.get("class_name"):
                data["class_name"] = f"{data.get('department', '')} - {data.get('section', '')}".strip(" -")
            return data

    def get_all_students(self) -> List[Dict[str, Any]]:
        """Retrieves all registered student records."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM students ORDER BY student_id ASC;")
            results = []
            for row in cursor.fetchall():
                data = dict(row)
                if not data.get("register_no"):
                    data["register_no"] = data["student_id"]
                if not data.get("class_name"):
                    data["class_name"] = f"{data.get('department', '')} - {data.get('section', '')}".strip(" -")
                results.append(data)
            return results

    def log_enrollment_event(
        self,
        event_type: str,
        student_id: Optional[str],
        details: str,
        status: str = "SUCCESS"
    ) -> int:
        """Logs an enrollment audit event."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO enrollment_events (event_type, student_id, details, status)
                VALUES (?, ?, ?, ?);
            """, (event_type, student_id, details, status))
            conn.commit()
            return cursor.lastrowid

    def delete_student(self, student_id: str) -> bool:
        """Deletes a student and their associated embeddings (via cascade)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM students WHERE student_id = ?;", (student_id,))
            conn.commit()
            return cursor.rowcount > 0

    def add_embedding(
        self,
        student_id: str,
        vector: np.ndarray,
        quality_score: float = 1.0,
        sample_label: str = "frontal",
        model_version: str = "w600k_mbf"
    ) -> int:
        """
        Stores an embedding vector as a binary BLOB associated with a student.
        Returns the inserted embedding_id.
        """
        if vector is None or not isinstance(vector, np.ndarray):
            raise ValueError("Vector must be a non-null numpy.ndarray.")

        # Ensure vector is contiguous float32 1D array
        vec_1d = np.ascontiguousarray(vector.flatten(), dtype=np.float32)
        blob = vec_1d.tobytes()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO embeddings (student_id, vector_blob, quality_score, sample_label, model_version)
                VALUES (?, ?, ?, ?, ?);
            """, (student_id, blob, quality_score, sample_label, model_version))
            conn.commit()
            return cursor.lastrowid

    def get_embeddings_for_student(self, student_id: str) -> List[Dict[str, Any]]:
        """Retrieves all embeddings for a specific student."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT embedding_id, student_id, vector_blob, quality_score, sample_label, model_version, created_at
                FROM embeddings WHERE student_id = ? ORDER BY embedding_id ASC;
            """, (student_id,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["vector"] = np.frombuffer(item["vector_blob"], dtype=np.float32)
                results.append(item)
            return results

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Retrieves all embeddings stored in the database with deserialized vectors."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT e.embedding_id, e.student_id, s.student_name, e.vector_blob, 
                       e.quality_score, e.sample_label, e.model_version, e.created_at
                FROM embeddings e
                JOIN students s ON e.student_id = s.student_id
                ORDER BY e.embedding_id ASC;
            """)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["vector"] = np.frombuffer(item["vector_blob"], dtype=np.float32)
                results.append(item)
            return results

    def get_student_count(self) -> int:
        """Returns the total number of enrolled students."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM students;")
            return cursor.fetchone()[0]

    def get_embedding_count(self) -> int:
        """Returns the total number of stored embeddings."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM embeddings;")
            return cursor.fetchone()[0]

    # =========================================================================
    # Phase 8: Session Management Database Methods
    # =========================================================================

    def create_session(
        self,
        session_id: str,
        date: str,
        class_section: str,
        subject: str,
        planned_start_time: str,
        planned_end_time: str,
        status: str = "SCHEDULED"
    ) -> bool:
        """Creates a new instructional session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (
                    session_id, date, class_section, subject,
                    planned_start_time, planned_end_time, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
            """, (session_id, date, class_section, subject, planned_start_time, planned_end_time, status))
            conn.commit()
            return True

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves session details by session_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?;", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_session_status(
        self,
        session_id: str,
        status: str,
        actual_start_time: Optional[str] = None,
        actual_end_time: Optional[str] = None
    ) -> bool:
        """Updates session state and actual start/end timestamps."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "UPDATE sessions SET status = ?, updated_at = CURRENT_TIMESTAMP"
            params = [status]
            if actual_start_time is not None:
                query += ", actual_start_time = ?"
                params.append(actual_start_time)
            if actual_end_time is not None:
                query += ", actual_end_time = ?"
                params.append(actual_end_time)
            query += " WHERE session_id = ?;"
            params.append(session_id)

            cursor.execute(query, tuple(params))
            conn.commit()
            return cursor.rowcount > 0

    def get_active_session(self, class_section: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Returns the currently active session (optionally for a specific class/section)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if class_section:
                cursor.execute(
                    "SELECT * FROM sessions WHERE status = 'ACTIVE' AND class_section = ? ORDER BY actual_start_time DESC LIMIT 1;",
                    (class_section,)
                )
            else:
                cursor.execute(
                    "SELECT * FROM sessions WHERE status = 'ACTIVE' ORDER BY actual_start_time DESC LIMIT 1;"
                )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_sessions_by_date(self, date: str) -> List[Dict[str, Any]]:
        """Retrieves all sessions on a given date (YYYY-MM-DD)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE date = ? ORDER BY planned_start_time ASC;", (date,))
            return [dict(row) for row in cursor.fetchall()]

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Retrieves all sessions."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC;")
            return [dict(row) for row in cursor.fetchall()]

    def get_unclosed_sessions(self) -> List[Dict[str, Any]]:
        """Retrieves sessions in ACTIVE, PAUSED, or RECOVERING states (for startup recovery)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM sessions WHERE status IN ('ACTIVE', 'PAUSED', 'RECOVERING') ORDER BY created_at ASC;"
            )
            return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Phase 8: Attendance Records Database Methods
    # =========================================================================

    def record_or_update_attendance(
        self,
        session_id: str,
        student_id: str,
        status: str,
        first_seen: str,
        last_seen: str,
        track_id: Optional[int] = None,
        similarity: float = 0.0
    ) -> Tuple[bool, bool]:
        """
        Atomically records attendance or updates existing record for a student in a session.
        Guarantees:
        - Exactly ONE record per (session_id, student_id)
        - first_seen, initial_similarity, and first_track_id are NEVER overwritten
        - last_seen, latest_similarity, and last_track_id are updated
        - status PRESENT is NEVER demoted to LATE

        Returns:
            (success: bool, is_new: bool)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Check if record already exists
            cursor.execute(
                "SELECT attendance_id, status FROM attendance WHERE session_id = ? AND student_id = ?;",
                (session_id, student_id)
            )
            existing = cursor.fetchone()

            if existing is None:
                # Insert fresh record
                cursor.execute("""
                    INSERT INTO attendance (
                        session_id, student_id, status,
                        first_seen, last_seen,
                        first_track_id, last_track_id,
                        initial_similarity, latest_similarity,
                        marked_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
                """, (
                    session_id, student_id, status,
                    first_seen, last_seen,
                    track_id, track_id,
                    similarity, similarity
                ))
                conn.commit()
                return True, True
            else:
                # Update existing record: retain original first_seen and status if already PRESENT
                current_status = existing["status"]
                final_status = "PRESENT" if current_status == "PRESENT" else status

                cursor.execute("""
                    UPDATE attendance SET
                        last_seen = ?,
                        latest_similarity = ?,
                        last_track_id = ?,
                        status = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ? AND student_id = ?;
                """, (last_seen, similarity, track_id, final_status, session_id, student_id))
                conn.commit()
                return True, False

    def get_attendance_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves all attendance records for a specific session joined with student names."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT a.attendance_id, a.session_id, a.student_id, s.student_name,
                       a.status, a.first_seen, a.last_seen, a.first_track_id, a.last_track_id,
                       a.initial_similarity, a.latest_similarity, a.marked_at, a.updated_at
                FROM attendance a
                LEFT JOIN students s ON a.student_id = s.student_id
                WHERE a.session_id = ?
                ORDER BY a.first_seen ASC;
            """, (session_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_attendance_record(self, session_id: str, student_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single attendance record for a student in a session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT a.attendance_id, a.session_id, a.student_id, s.student_name,
                       a.status, a.first_seen, a.last_seen, a.first_track_id, a.last_track_id,
                       a.initial_similarity, a.latest_similarity, a.marked_at, a.updated_at
                FROM attendance a
                LEFT JOIN students s ON a.student_id = s.student_id
                WHERE a.session_id = ? AND a.student_id = ?;
            """, (session_id, student_id))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_student_attendance_history(self, student_id: str) -> List[Dict[str, Any]]:
        """Retrieves complete attendance history across all sessions for a student."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT a.attendance_id, a.session_id, sess.date, sess.class_section, sess.subject,
                       a.status, a.first_seen, a.last_seen, a.initial_similarity, a.latest_similarity
                FROM attendance a
                JOIN sessions sess ON a.session_id = sess.session_id
                WHERE a.student_id = ?
                ORDER BY sess.date DESC, sess.planned_start_time DESC;
            """, (student_id,))
            return [dict(row) for row in cursor.fetchall()]

    def clear_all(self):
        """Clears all records from attendance, sessions, embeddings, students, and metadata."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM attendance;")
            cursor.execute("DELETE FROM sessions;")
            cursor.execute("DELETE FROM embeddings;")
            cursor.execute("DELETE FROM students;")
            cursor.execute("DELETE FROM model_metadata;")
            conn.commit()

