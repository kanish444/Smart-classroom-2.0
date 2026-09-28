import os
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from loguru import logger
from config.settings import get_settings


class NewEnrollmentDatabase:
    """
    Dedicated, isolated SQLite database manager for Phase 10 One-By-One Face Enrollment.
    Completely independent from any previous student documents, spreadsheets, or legacy databases.
    Manages:
    - enrolled_students: (student_id, register_number, name, class, department, section, created_at, updated_at, enrollment_status)
    - enrolled_embeddings: (embedding_id, student_id, register_number, vector_blob, sample_index, quality_score, created_at)
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            self.db_path = db_path
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.db_path = os.path.join(base_dir, "database", "new_enrollment.sqlite")

        # Ensure parent directory exists
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Returns an active SQLite connection with row factory enabled and foreign keys enforced."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        """Initializes tables for new enrolled students and their face embeddings."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS enrolled_students (
                    student_id TEXT PRIMARY KEY,
                    register_number TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    class TEXT NOT NULL,
                    department TEXT NOT NULL,
                    section TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    enrollment_status TEXT DEFAULT 'COMPLETED'
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS enrolled_embeddings (
                    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    register_number TEXT NOT NULL,
                    vector_blob BLOB NOT NULL,
                    sample_index INTEGER NOT NULL,
                    quality_score REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES enrolled_students(student_id) ON DELETE CASCADE
                );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_enrolled_reg_no ON enrolled_students(register_number);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_enrolled_emb_student ON enrolled_embeddings(student_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_enrolled_emb_reg ON enrolled_embeddings(register_number);")
            conn.commit()

    def check_register_number_exists(self, register_number: str) -> bool:
        """Checks if a Register Number is already enrolled."""
        if not register_number:
            return False
        clean_reg = str(register_number).strip()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM enrolled_students WHERE register_number = ? LIMIT 1;", (clean_reg,))
            return cursor.fetchone() is not None

    def get_student_by_register_number(self, register_number: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single student profile by Register Number."""
        clean_reg = str(register_number).strip()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM enrolled_students WHERE register_number = ?;", (clean_reg,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_student_by_id(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a student profile by student_id or register_number."""
        clean_id = str(student_id).strip()
        reg_query = clean_id[4:] if clean_id.startswith("STU_") else clean_id
        stu_query = f"STU_{clean_id}" if not clean_id.startswith("STU_") else clean_id
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM enrolled_students WHERE student_id = ? OR register_number = ? OR student_id = ? OR register_number = ? LIMIT 1;",
                (clean_id, clean_id, reg_query, stu_query)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_students(self) -> List[Dict[str, Any]]:
        """Retrieves all registered students in the new enrollment database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM enrolled_students ORDER BY created_at ASC;")
            return [dict(row) for row in cursor.fetchall()]

    def get_embeddings_for_student(self, student_id: str) -> List[Dict[str, Any]]:
        """Retrieves all embeddings for a specific student, with deserialized numpy vectors."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT embedding_id, student_id, register_number, vector_blob, sample_index, quality_score, created_at
                FROM enrolled_embeddings
                WHERE student_id = ?
                ORDER BY sample_index ASC;
            """, (student_id,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["vector"] = np.frombuffer(item["vector_blob"], dtype=np.float32)
                results.append(item)
            return results

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Retrieves all embeddings in the new enrollment database with deserialized vectors."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT e.embedding_id, e.student_id, e.register_number, s.name, s.class, s.department, s.section,
                       e.vector_blob, e.sample_index, e.quality_score, e.created_at
                FROM enrolled_embeddings e
                JOIN enrolled_students s ON e.student_id = s.student_id
                ORDER BY e.embedding_id ASC;
            """)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["vector"] = np.frombuffer(item["vector_blob"], dtype=np.float32)
                results.append(item)
            return results

    def delete_student(self, student_id: str) -> bool:
        """Deletes a student and their associated embeddings (via ON DELETE CASCADE)."""
        clean_id = str(student_id).strip()
        reg_query = clean_id[4:] if clean_id.startswith("STU_") else clean_id
        stu_query = f"STU_{clean_id}" if not clean_id.startswith("STU_") else clean_id
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM enrolled_students WHERE student_id = ? OR register_number = ? OR student_id = ? OR register_number = ?;",
                (clean_id, clean_id, reg_query, stu_query)
            )
            conn.commit()
            deleted = cursor.rowcount > 0

        # Also purge from smartclass.sqlite students table
        try:
            from database.db_manager import DatabaseManager
            db_mgr = DatabaseManager()
            db_mgr.delete_student(clean_id)
            if reg_query != clean_id:
                db_mgr.delete_student(reg_query)
        except Exception:
            pass

        return deleted

    def get_student_count(self) -> int:
        """Returns the total number of enrolled students."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM enrolled_students;")
            return cursor.fetchone()[0]

    def get_embedding_count(self) -> int:
        """Returns the total number of stored embeddings."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM enrolled_embeddings;")
            return cursor.fetchone()[0]

    def save_enrollment_atomic(
        self,
        student_id: str,
        register_number: str,
        name: str,
        class_name: str,
        department: str,
        section: str,
        embeddings: List[np.ndarray],
        quality_scores: Optional[List[float]] = None
    ) -> List[int]:
        """
        Atomically saves a new student and their multiple face embeddings.
        Guarantees that if duplicate register_number or any insert fails,
        the entire transaction is rolled back with no partial enrollment.
        
        Returns:
            List of generated embedding_ids.
        """
        clean_reg = str(register_number).strip()
        clean_name = str(name).strip()
        clean_class = str(class_name).strip()
        clean_dept = str(department).strip()
        clean_sec = str(section).strip()

        if not clean_reg:
            raise ValueError("Register Number cannot be empty.")
        if not clean_name:
            raise ValueError("Student Name cannot be empty.")
        if not clean_class:
            raise ValueError("Class cannot be empty.")
        if not clean_dept:
            raise ValueError("Department cannot be empty.")
        if not clean_sec:
            raise ValueError("Section cannot be empty.")
        if not embeddings or len(embeddings) == 0:
            raise ValueError("At least one face embedding is required.")

        scores = quality_scores or [1.0] * len(embeddings)
        if len(scores) != len(embeddings):
            scores = [1.0] * len(embeddings)

        inserted_embedding_ids: List[int] = []

        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            # Verify duplicate register number
            cursor.execute("SELECT 1 FROM enrolled_students WHERE register_number = ?;", (clean_reg,))
            if cursor.fetchone() is not None:
                raise ValueError(f"Register Number '{clean_reg}' is already enrolled.")

            # Insert student
            cursor.execute("""
                INSERT INTO enrolled_students (
                    student_id, register_number, name, class, department, section, enrollment_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'COMPLETED', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
            """, (student_id, clean_reg, clean_name, clean_class, clean_dept, clean_sec))

            # Insert embeddings
            for idx, (emb, q_score) in enumerate(zip(embeddings, scores)):
                vec_1d = np.ascontiguousarray(emb.flatten(), dtype=np.float32)
                blob = vec_1d.tobytes()
                cursor.execute("""
                    INSERT INTO enrolled_embeddings (
                        student_id, register_number, vector_blob, sample_index, quality_score, created_at
                    ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
                """, (student_id, clean_reg, blob, idx + 1, float(q_score)))
                inserted_embedding_ids.append(cursor.lastrowid)

            conn.commit()
            logger.info(f"Atomically enrolled student '{clean_name}' ({clean_reg}) with {len(inserted_embedding_ids)} embeddings.")
            return inserted_embedding_ids

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to atomically enroll student {clean_reg}: {e}")
            raise e
        finally:
            conn.close()
