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

    def __init__(self, db_path: Optional[str] = None, new_db_path: Optional[str] = None):
        self.settings = get_settings()
        self.db_path = db_path or self.settings.db_path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.new_db_path = new_db_path or os.path.join(base_dir, "database", "new_enrollment.sqlite")

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

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS classrooms (
                    classroom_id TEXT PRIMARY KEY,
                    classroom_name TEXT NOT NULL,
                    department TEXT NOT NULL,
                    year TEXT NOT NULL,
                    section TEXT NOT NULL,
                    assigned_advisor_id TEXT,
                    assigned_advisor_name TEXT,
                    camera_source TEXT DEFAULT 'pc',
                    camera_url TEXT,
                    camera_status TEXT DEFAULT 'connected',
                    esp32_device_id TEXT,
                    esp32_status TEXT DEFAULT 'disconnected',
                    ai_pipeline_status TEXT DEFAULT 'running',
                    faiss_status TEXT DEFAULT 'ready',
                    attendance_status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Bootstrap default classroom if table is newly created
            cursor.execute("SELECT COUNT(*) FROM classrooms;")
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    INSERT INTO classrooms (
                        classroom_id, classroom_name, department, year, section,
                        assigned_advisor_id, assigned_advisor_name, camera_source,
                        camera_status, esp32_device_id, esp32_status,
                        ai_pipeline_status, faiss_status, attendance_status
                    ) VALUES (
                        'AIDS-B', 'AIDS-B Smart Classroom', 'AI&DS', '3rd Year', 'B',
                        'ADV001', 'Prof. Ramesh Kumar', 'pc',
                        'connected', 'ESP32-AIDS-B', 'disconnected',
                        'running', 'ready', 'active'
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

    def sync_enrolled_students(self) -> int:
        """
        Synchronizes smartclass.sqlite students table with authoritative new_enrollment.sqlite.
        Guarantees:
        - Only students currently enrolled in new_enrollment.sqlite exist in smartclass.sqlite
        - Stale/test student records (e.g. STU_DTEST_*) are purged
        - Foreign keys for attendance records are cleanly preserved
        """
        new_db_path = self.new_db_path
        if not os.path.exists(new_db_path):
            return 0

        synced_count = 0
        try:
            with sqlite3.connect(new_db_path) as nconn:
                nconn.row_factory = sqlite3.Row
                ncur = nconn.cursor()
                ncur.execute("SELECT * FROM enrolled_students;")
                enrolled = [dict(r) for r in ncur.fetchall()]

            valid_ids = [s["student_id"] for s in enrolled]
            valid_regs = [s["register_number"] for s in enrolled]

            with self.get_connection() as conn:
                cur = conn.cursor()
                # Remove stale students not in active enrollment
                if valid_ids:
                    all_valid = list(set(valid_ids + valid_regs))
                    placeholders = ",".join(["?"] * len(all_valid))
                    cur.execute(f"DELETE FROM students WHERE student_id NOT IN ({placeholders}) AND register_no NOT IN ({placeholders});", all_valid + all_valid)
                else:
                    cur.execute("DELETE FROM students;")

                # Upsert active enrolled students
                for s in enrolled:
                    sid = s["student_id"]
                    reg = s["register_number"]
                    name = s["name"]
                    dept = s.get("department", "")
                    sec = s.get("section", "")
                    cls_name = s.get("class") or f"{dept} - {sec}".strip(" -")
                    cur.execute("""
                        INSERT INTO students (student_id, student_name, department, section, status, register_no, class_name, updated_at)
                        VALUES (?, ?, ?, ?, 'active', ?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(student_id) DO UPDATE SET
                            student_name = excluded.student_name,
                            department = excluded.department,
                            section = excluded.section,
                            status = excluded.status,
                            register_no = excluded.register_no,
                            class_name = excluded.class_name,
                            updated_at = CURRENT_TIMESTAMP;
                    """, (sid, name, dept, sec, reg, cls_name))
                    synced_count += 1
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to sync enrolled students: {e}")

        return synced_count

    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single student profile by ID or Register Number from the authoritative registry.
        Seamlessly resolves both raw register number and any prefixed aliases.
        """
        clean_id = str(student_id).strip()
        reg_query = clean_id[4:] if clean_id.startswith("STU_") else clean_id
        stu_query = f"STU_{clean_id}" if not clean_id.startswith("STU_") else clean_id

        new_db_path = self.new_db_path
        if os.path.exists(new_db_path):
            try:
                with sqlite3.connect(new_db_path) as nconn:
                    nconn.row_factory = sqlite3.Row
                    ncur = nconn.cursor()
                    ncur.execute(
                        "SELECT * FROM enrolled_students WHERE student_id = ? OR register_number = ? OR student_id = ? OR register_number = ? LIMIT 1;",
                        (clean_id, clean_id, reg_query, stu_query)
                    )
                    nrow = ncur.fetchone()
                    if nrow:
                        ndata = dict(nrow)
                        sid = ndata.get("student_id") or ndata.get("register_number")
                        reg = ndata.get("register_number") or sid
                        return {
                            "student_id": sid,
                            "student_name": ndata["name"],
                            "register_no": reg,
                            "department": ndata.get("department", ""),
                            "section": ndata.get("section", ""),
                            "class_name": ndata.get("class", ""),
                            "status": ndata.get("enrollment_status", "active")
                        }
            except Exception as e:
                logger.warning(f"Authoritative student lookup in new_enrollment.sqlite failed: {e}")

        # Fallback to local students table
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM students WHERE student_id = ? OR register_no = ? OR student_id = ? LIMIT 1;",
                (clean_id, clean_id, reg_query)
            )
            row = cursor.fetchone()
            if row:
                data = dict(row)
                if not data.get("register_no"):
                    data["register_no"] = data["student_id"]
                if not data.get("class_name"):
                    data["class_name"] = f"{data.get('department', '')} - {data.get('section', '')}".strip(" -")
                return data

        return None

    def get_all_students(self) -> List[Dict[str, Any]]:
        """
        Retrieves all registered student records strictly from the authoritative new enrollment registry.
        Never loads un-enrolled legacy records, hardcoded students, or old test data.
        """
        new_db_path = self.new_db_path
        if os.path.exists(new_db_path):
            try:
                results = []
                with sqlite3.connect(new_db_path) as nconn:
                    nconn.row_factory = sqlite3.Row
                    ncur = nconn.cursor()
                    ncur.execute("SELECT * FROM enrolled_students ORDER BY name ASC;")
                    for nrow in ncur.fetchall():
                        ndata = dict(nrow)
                        sid = ndata.get("student_id") or ndata.get("register_number")
                        reg = ndata.get("register_number") or sid
                        results.append({
                            "student_id": sid,
                            "student_name": ndata.get("name"),
                            "register_no": reg,
                            "department": ndata.get("department", ""),
                            "section": ndata.get("section", ""),
                            "class_name": ndata.get("class", ""),
                            "status": ndata.get("enrollment_status", "active")
                        })
                return results
            except Exception as e:
                logger.error(f"Querying authoritative new_enrollment.sqlite: {e}")

        # Fallback to local students table only if new_enrollment does not exist
        results = []
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM students ORDER BY student_name ASC;")
                for row in cursor.fetchall():
                    data = dict(row)
                    if not data.get("register_no"):
                        data["register_no"] = data["student_id"]
                    if not data.get("class_name"):
                        data["class_name"] = f"{data.get('department', '')} - {data.get('section', '')}".strip(" -")
                    results.append(data)
        except Exception as e:
            logger.debug(f"Querying students table fallback: {e}")
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
        """Returns the total number of enrolled students strictly from the authoritative new enrollment registry."""
        new_db_path = self.new_db_path
        if os.path.exists(new_db_path):
            try:
                with sqlite3.connect(new_db_path) as nconn:
                    ncur = nconn.cursor()
                    ncur.execute("SELECT COUNT(*) FROM enrolled_students;")
                    return ncur.fetchone()[0]
            except Exception as e:
                logger.error(f"Error reading student count from new_enrollment.sqlite: {e}")

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM students;")
                return cursor.fetchone()[0]
        except Exception:
            return 0

    def get_embedding_count(self) -> int:
        """Returns the total number of stored embeddings strictly from the authoritative new enrollment registry."""
        new_db_path = self.new_db_path
        if os.path.exists(new_db_path):
            try:
                with sqlite3.connect(new_db_path) as nconn:
                    ncur = nconn.cursor()
                    ncur.execute("SELECT COUNT(*) FROM enrolled_embeddings;")
                    return ncur.fetchone()[0]
            except Exception as e:
                logger.error(f"Error reading embedding count from new_enrollment.sqlite: {e}")

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM embeddings;")
                return cursor.fetchone()[0]
        except Exception:
            return 0

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

            # Ensure student exists in students table to satisfy foreign key constraint
            cursor.execute("SELECT 1 FROM students WHERE student_id = ?;", (student_id,))
            if cursor.fetchone() is None:
                st = self.get_student(student_id)
                if st:
                    cls_name = st.get("class_name") or f"{st.get('department', '')} - {st.get('section', '')}".strip(" -")
                    cursor.execute("""
                        INSERT INTO students (student_id, student_name, department, section, status, register_no, class_name, updated_at)
                        VALUES (?, ?, ?, ?, 'active', ?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(student_id) DO UPDATE SET
                            student_name = excluded.student_name,
                            department = excluded.department,
                            section = excluded.section,
                            status = excluded.status,
                            register_no = excluded.register_no,
                            class_name = excluded.class_name,
                            updated_at = CURRENT_TIMESTAMP;
                    """, (student_id, st["student_name"], st.get("department", ""), st.get("section", ""), st.get("register_no", student_id), cls_name))
                    conn.commit()

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
            rows = [dict(row) for row in cursor.fetchall()]

        # Ensure student_name is always resolved from authoritative student profile
        for r in rows:
            if not r.get("student_name"):
                st = self.get_student(r["student_id"])
                if st:
                    r["student_name"] = st["student_name"]
                else:
                    r["student_name"] = r["student_id"]
        return rows

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

    # =========================================================================
    # Classroom Management & Device Mapping Methods
    # =========================================================================

    def get_all_classrooms(self) -> List[Dict[str, Any]]:
        """Retrieves all registered classrooms and their mapping configurations."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM classrooms ORDER BY classroom_id ASC;")
            return [dict(row) for row in cursor.fetchall()]

    def get_classroom_by_id(self, classroom_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single classroom by its classroom_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM classrooms WHERE classroom_id = ? LIMIT 1;", (classroom_id.strip(),))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_classroom_by_advisor_id(self, advisor_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the classroom mapped to a specific advisor."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM classrooms WHERE assigned_advisor_id = ? LIMIT 1;", (advisor_id.strip(),))
            row = cursor.fetchone()
            return dict(row) if row else None

    def upsert_classroom(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Creates or updates a classroom entity and its hardware mappings."""
        cid = data["classroom_id"].strip()
        cname = data.get("classroom_name") or f"{cid} Smart Classroom"
        dept = data.get("department", "AI&DS")
        year = data.get("year", "3rd Year")
        sec = data.get("section", "B")
        adv_id = data.get("assigned_advisor_id")
        adv_name = data.get("assigned_advisor_name")
        cam_src = data.get("camera_source", "none")
        cam_url = data.get("camera_url")
        cam_status = data.get("camera_status", "connected" if cam_src != "none" else "disconnected")
        esp_id = data.get("esp32_device_id")
        esp_status = data.get("esp32_status", "disconnected")
        ai_stat = data.get("ai_pipeline_status", "running")
        faiss_stat = data.get("faiss_status", "ready")
        att_stat = data.get("attendance_status", "active")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO classrooms (
                    classroom_id, classroom_name, department, year, section,
                    assigned_advisor_id, assigned_advisor_name, camera_source,
                    camera_url, camera_status, esp32_device_id, esp32_status,
                    ai_pipeline_status, faiss_status, attendance_status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(classroom_id) DO UPDATE SET
                    classroom_name = excluded.classroom_name,
                    department = excluded.department,
                    year = excluded.year,
                    section = excluded.section,
                    assigned_advisor_id = excluded.assigned_advisor_id,
                    assigned_advisor_name = excluded.assigned_advisor_name,
                    camera_source = excluded.camera_source,
                    camera_url = excluded.camera_url,
                    camera_status = excluded.camera_status,
                    esp32_device_id = excluded.esp32_device_id,
                    esp32_status = excluded.esp32_status,
                    ai_pipeline_status = excluded.ai_pipeline_status,
                    faiss_status = excluded.faiss_status,
                    attendance_status = excluded.attendance_status,
                    updated_at = CURRENT_TIMESTAMP;
            """, (
                cid, cname, dept, year, sec,
                adv_id, adv_name, cam_src,
                cam_url, cam_status, esp_id, esp_status,
                ai_stat, faiss_stat, att_stat
            ))
            conn.commit()

        return self.get_classroom_by_id(cid)

    def delete_classroom(self, classroom_id: str) -> bool:
        """Deletes a classroom record."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM classrooms WHERE classroom_id = ?;", (classroom_id.strip(),))
            conn.commit()
            return cursor.rowcount > 0

