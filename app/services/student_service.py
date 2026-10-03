import os
import shutil
import sqlite3
from typing import Dict, Any, Optional, Set, List
from loguru import logger
from fastapi import HTTPException, status

from database.new_enrollment_db import NewEnrollmentDatabase
from database.db_manager import DatabaseManager
from core.schemas import RecognitionStatus


class StudentService:
    """
    Service layer for student management operations.
    Handles authoritative cross-database deletion, face biometric vector purging,
    FAISS index reconstruction, active track cache invalidation, and attendance cleanup.
    """

    @classmethod
    def delete_student(cls, identifier: str, state: Any) -> Dict[str, Any]:
        """
        Permanently deletes a student across all subsystems:
        1. Student profile: removed from new_enrollment.sqlite and smartclass.sqlite
        2. Face enrollment data: local photo directories and enrollment audit events purged
        3. Face embeddings: removed from SQLite enrolled_embeddings and embeddings tables
        4. FAISS index: rebuilt without the deleted student, zero orphan vectors, saved to disk
        5. Live tracking: temporal stabilizer and active telemetry caches invalidated
        6. Attendance records: student-specific records removed
        7. Deduplication cache: student events cleared from in-memory engine

        Args:
            identifier: Register number or Student ID.
            state: AppState instance.

        Returns:
            Dictionary with deletion summary.

        Raises:
            HTTPException: 404 if student not found, 500 on database error.
        """
        clean_id = str(identifier).strip()
        if not clean_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Identifier cannot be empty.")

        reg_query = clean_id[4:] if clean_id.startswith("STU_") else clean_id
        stu_query = f"STU_{clean_id}" if not clean_id.startswith("STU_") else clean_id

        # 1. Locate student across databases
        new_db_path = getattr(state.db, "new_db_path", None)
        new_db = NewEnrollmentDatabase(db_path=new_db_path)
        enrolled_row = None
        main_row = None

        with new_db.get_connection() as nconn:
            ncur = nconn.cursor()
            ncur.execute(
                "SELECT * FROM enrolled_students WHERE student_id = ? OR register_number = ? OR student_id = ? OR register_number = ?;",
                (clean_id, clean_id, reg_query, stu_query)
            )
            r = ncur.fetchone()
            if r:
                enrolled_row = dict(r)

        with state.db.get_connection() as mconn:
            mcur = mconn.cursor()
            mcur.execute(
                "SELECT * FROM students WHERE student_id = ? OR register_no = ? OR student_id = ? OR register_no = ?;",
                (clean_id, clean_id, reg_query, stu_query)
            )
            r = mcur.fetchone()
            if r:
                main_row = dict(r)

        if not enrolled_row and not main_row:
            logger.warning(f"Student deletion requested for non-existent identifier: '{clean_id}'")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student with register number/ID '{clean_id}' not found."
            )

        # Extract authoritative details
        student_name = "Unknown"
        reg_no = clean_id
        stu_id = clean_id
        department = "AI&DS"
        year = "3rd Year"
        section = "B"

        if enrolled_row:
            stu_id = enrolled_row.get("student_id") or stu_id
            reg_no = enrolled_row.get("register_number") or reg_no
            student_name = enrolled_row.get("name") or student_name
            department = enrolled_row.get("department") or department
            year = enrolled_row.get("class") or year
            section = enrolled_row.get("section") or section

        if main_row:
            stu_id = main_row.get("student_id") or stu_id
            reg_no = main_row.get("register_no") or reg_no
            student_name = main_row.get("student_name") or student_name
            department = main_row.get("department") or department
            year = main_row.get("class_name") or year
            section = main_row.get("section") or section

        # All possible identifiers to purge from relations
        target_ids: Set[str] = {
            str(stu_id).strip(),
            str(reg_no).strip(),
            clean_id,
            reg_query,
            stu_query
        }
        target_ids.discard("")

        logger.info(f"Initiating comprehensive deletion for student '{student_name}' (Reg: {reg_no}, ID: {stu_id})")

        # 2. Purge from new_enrollment.sqlite (enrolled_students & enrolled_embeddings)
        with new_db.get_connection() as nconn:
            ncur = nconn.cursor()
            for tid in target_ids:
                ncur.execute("DELETE FROM enrolled_embeddings WHERE student_id = ? OR register_number = ?;", (tid, tid))
                ncur.execute("DELETE FROM enrolled_students WHERE student_id = ? OR register_number = ?;", (tid, tid))
            nconn.commit()

        # 3. Purge from smartclass.sqlite (students, attendance, embeddings, enrollment events)
        with state.db.get_connection() as mconn:
            mcur = mconn.cursor()
            for tid in target_ids:
                mcur.execute("DELETE FROM attendance WHERE student_id = ?;", (tid,))
                mcur.execute("DELETE FROM embeddings WHERE student_id = ?;", (tid,))
                mcur.execute("DELETE FROM enrollment_events WHERE student_id = ?;", (tid,))
                mcur.execute("DELETE FROM enrollment_sources WHERE student_id = ?;", (tid,))
                mcur.execute("DELETE FROM students WHERE student_id = ? OR register_no = ?;", (tid, tid))
            mconn.commit()

        # 4. Remove local enrollment photo directory if present
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for tid in target_ids:
            photo_dir = os.path.join(base_dir, "data", "enrollment", tid)
            if os.path.isdir(photo_dir):
                try:
                    shutil.rmtree(photo_dir, ignore_errors=True)
                    logger.info(f"Removed enrollment photos directory: {photo_dir}")
                except Exception as pe:
                    logger.warning(f"Could not delete photo directory {photo_dir}: {pe}")

        # 5. Clean and rebuild FAISS vector database to ensure ZERO orphan vectors
        v_store = state.get_vector_store()
        remaining_embs = new_db.get_all_embeddings()

        # Completely reset FAISS index and reload only surviving embeddings
        v_store.clear()
        for rec in remaining_embs:
            emb_id = rec["embedding_id"]
            vec = rec["vector"]
            meta = {
                "embedding_id": emb_id,
                "student_id": rec["student_id"],
                "student_name": rec.get("name") or rec.get("student_name") or "Unknown",
                "register_number": rec.get("register_number") or rec["student_id"],
                "class": rec.get("class", ""),
                "department": rec.get("department", ""),
                "section": rec.get("section", ""),
                "quality_score": float(rec.get("quality_score", 1.0))
            }
            v_store.add_vector(emb_id, vec, meta)

        v_store.save_index()
        state.sync_active_recognition()
        logger.info(f"FAISS index reconstructed: {v_store.total_vectors} vectors remaining (zero orphan vectors).")

        # 6. Invalidate recognition tracking data and caches
        if state.tracking_pipeline and hasattr(state.tracking_pipeline, "stabilizer") and state.tracking_pipeline.stabilizer:
            try:
                state.tracking_pipeline.stabilizer.invalidate_student(stu_id, reg_no)
            except Exception as se:
                logger.warning(f"Temporal stabilizer invalidation warning: {se}")

        # Invalidate active telemetry tracks
        with state._lock:
            for t in getattr(state, "latest_tracks", []):
                t_sid = str(getattr(t, "stable_student_id", "") or "").strip()
                if t_sid in target_ids:
                    t.stable_student_id = None
                    t.display_name = "Unknown"
                    t.display_status = "UNKNOWN"

        # 7. Invalidate in-memory AttendanceEngine deduplication buffer
        if state.attendance_engine and hasattr(state.attendance_engine, "_dedup_cache"):
            try:
                for k in list(state.attendance_engine._dedup_cache.keys()):
                    if str(k[1]).strip() in target_ids:
                        del state.attendance_engine._dedup_cache[k]
            except Exception as de:
                logger.warning(f"Attendance dedup cache cleanup warning: {de}")

        logger.info(f"Student '{student_name}' ({reg_no}) successfully and completely deleted.")

        return {
            "success": True,
            "deleted": True,
            "student_id": stu_id,
            "register_number": reg_no,
            "student_name": student_name,
            "department": department,
            "year": year,
            "section": section,
            "remaining_vectors": v_store.total_vectors,
            "message": f"Student '{student_name}' ({reg_no}) deleted successfully."
        }
