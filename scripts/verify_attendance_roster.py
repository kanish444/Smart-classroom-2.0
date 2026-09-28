import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import json
import sqlite3
import faiss
from database.db_manager import DatabaseManager
from database.new_enrollment_db import NewEnrollmentDatabase
from attendance.session_manager import SessionManager
from attendance.attendance_engine import AttendanceEngine

def verify_system_state():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 60)
    print("SMARTCLASS VISION AI - ATTENDANCE & ENROLLMENT VERIFICATION")
    print("=" * 60)

    # 1. Authoritative Enrollment Database
    enroll_db = NewEnrollmentDatabase()
    enrolled_students = enroll_db.get_all_students()
    enroll_count = len(enrolled_students)

    print("\n[1] CURRENT ENROLLED STUDENTS (new_enrollment.sqlite):")
    for s in enrolled_students:
        print(f"  - Register Number: {s['register_number']}")
        print(f"    Name:            {s['name']}")
        print(f"    Student ID:      {s['student_id']}")
        print(f"    Class/Dept/Sec:  {s['class']} | {s['department']} | {s['section']}")
    print(f"  Total Enrolled Students: {enroll_count}")

    # 2. FAISS Vector Store
    faiss_bin = os.path.join(base_dir, "database", "faiss_index.bin")
    faiss_meta = os.path.join(base_dir, "database", "faiss_index.bin.meta.json")
    vector_count = 0
    meta_count = 0
    if os.path.exists(faiss_bin):
        index = faiss.read_index(faiss_bin)
        vector_count = index.ntotal
    if os.path.exists(faiss_meta):
        with open(faiss_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
            meta_count = len(meta)

    print(f"\n[2] FAISS VECTOR STORE:")
    print(f"  Total FAISS Vectors:      {vector_count}")
    print(f"  Total Vector Metadata:    {meta_count}")

    # 3. DatabaseManager Student Table & Sync
    db = DatabaseManager()
    db.sync_enrolled_students()
    db_students = db.get_all_students()
    db_count = db.get_student_count()
    db_emb_count = db.get_embedding_count()

    print(f"\n[3] DATABASE MANAGER AUTHORITATIVE VIEW:")
    print(f"  get_student_count():      {db_count}")
    print(f"  get_embedding_count():    {db_emb_count}")
    print(f"  get_all_students() count: {len(db_students)}")

    # 4. Attendance Roster (Simulation with test session)
    sm = SessionManager(db)
    engine = AttendanceEngine(db, sm)

    test_sess_id = "SESS_VERIFY_TEMP"
    # Ensure clean slate for test session
    try:
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM attendance WHERE session_id = ?;", (test_sess_id,))
            cur.execute("DELETE FROM sessions WHERE session_id = ?;", (test_sess_id,))
            conn.commit()
    except Exception:
        pass

    sess = sm.create_session(test_sess_id, "2026-09-25", "ALL", "Verification Test", "09:00", "12:00")
    report = engine.generate_session_report(sess.session_id)
    roster_student_count = report.total_enrolled

    print(f"\n[4] ATTENDANCE ROSTER INITIALIZATION:")
    print(f"  Session ID:               {sess.session_id}")
    print(f"  Attendance Roster Total:  {roster_student_count}")
    print(f"  Initial Present:          {report.present_count}")
    print(f"  Initial Late:             {report.late_count}")
    print(f"  Initial Not Seen:         {report.not_seen_count}")
    print(f"  Not Seen Student IDs:     {report.not_seen_students}")

    # Cleanup test session
    try:
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM attendance WHERE session_id = ?;", (test_sess_id,))
            cur.execute("DELETE FROM sessions WHERE session_id = ?;", (test_sess_id,))
            conn.commit()
    except Exception:
        pass

    # 5. Verification Checks
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY:")
    print("=" * 60)
    print(f"Enrollment Students Count: {enroll_count}")
    print(f"FAISS Vectors Count:       {vector_count}")
    print(f"Database Manager Count:    {db_count}")
    print(f"Attendance Roster Count:   {roster_student_count}")

    is_consistent = (enroll_count == db_count == roster_student_count)
    if is_consistent:
        print("\n>>> STATUS: SUCCESS! All counts are 100% consistent across components.")
    else:
        print("\n>>> STATUS: MISMATCH DETECTED!")
    print("=" * 60)
    return is_consistent

if __name__ == "__main__":
    verify_system_state()
