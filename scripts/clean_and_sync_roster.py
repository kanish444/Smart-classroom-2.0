import os
import sqlite3
import json
import faiss
from loguru import logger

def clean_and_sync():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    new_db_path = os.path.join(base_dir, "database", "new_enrollment.sqlite")
    main_db_path = os.path.join(base_dir, "database", "smartclass.sqlite")
    faiss_meta_path = os.path.join(base_dir, "database", "faiss_index.bin.meta.json")

    print("=== STEP 1: CLEAN AND NORMALIZE NEW_ENROLLMENT.SQLITE ===")
    if not os.path.exists(new_db_path):
        print(f"Error: {new_db_path} does not exist.")
        return

    nconn = sqlite3.connect(new_db_path)
    ncur = nconn.cursor()

    # Normalize student_id = register_number in enrolled_students
    ncur.execute("SELECT student_id, register_number, name FROM enrolled_students;")
    students = ncur.fetchall()
    print("Found students in new_enrollment.sqlite:", students)

    # Disable foreign keys temporarily for normalization update
    ncur.execute("PRAGMA foreign_keys = OFF;")
    for sid, reg, name in students:
        clean_reg = str(reg).strip()
        clean_name = str(name).strip()
        ncur.execute(
            "UPDATE enrolled_students SET student_id = ?, register_number = ?, name = ? WHERE student_id = ?;",
            (clean_reg, clean_reg, clean_name, sid)
        )
        ncur.execute(
            "UPDATE enrolled_embeddings SET student_id = ?, register_number = ? WHERE student_id = ?;",
            (clean_reg, clean_reg, sid)
        )
        print(f"Normalized student '{clean_name}': student_id='{clean_reg}', register_number='{clean_reg}'")

    ncur.execute("PRAGMA foreign_keys = ON;")
    nconn.commit()

    ncur.execute("SELECT student_id, register_number, name, class, department, section FROM enrolled_students;")
    active_enrolled = [dict(zip(["student_id", "register_number", "name", "class", "department", "section"], r)) for r in ncur.fetchall()]
    ncur.execute("SELECT COUNT(*) FROM enrolled_embeddings;")
    total_embeddings = ncur.fetchone()[0]
    nconn.close()

    print(f"Active enrolled students count: {len(active_enrolled)}")
    print(f"Total enrolled embeddings count: {total_embeddings}")

    print("\n=== STEP 2: UPDATE FAISS METADATA JSON ===")
    if os.path.exists(faiss_meta_path):
        with open(faiss_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        for vec_id, vmeta in meta.items():
            reg = str(vmeta.get("register_number", "")).strip()
            if reg:
                vmeta["student_id"] = reg
                vmeta["register_number"] = reg
        with open(faiss_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Updated FAISS metadata with normalized student_id for {len(meta)} vectors.")

    print("\n=== STEP 3: CLEAN AND SYNC SMARTCLASS.SQLITE ===")
    if os.path.exists(main_db_path):
        mconn = sqlite3.connect(main_db_path)
        mcur = mconn.cursor()

        # Remove stale test sessions (SESS_DTEST_*)
        mcur.execute("DELETE FROM attendance WHERE session_id LIKE 'SESS_DTEST_%';")
        mcur.execute("DELETE FROM sessions WHERE session_id LIKE 'SESS_DTEST_%';")
        print("Removed all stale SESS_DTEST_* sessions and attendance.")

        # Remove any student that is NOT in active_enrolled
        valid_ids = [s["student_id"] for s in active_enrolled]
        placeholders = ",".join(["?"] * len(valid_ids)) if valid_ids else "''"
        mcur.execute(f"DELETE FROM attendance WHERE student_id NOT IN ({placeholders});", valid_ids)
        mcur.execute(f"DELETE FROM students WHERE student_id NOT IN ({placeholders});", valid_ids)
        print(f"Removed stale students not in active enrollment.")

        # Sync active enrolled students into smartclass.sqlite students table
        for s in active_enrolled:
            cls_name = f"{s['department']} - {s['section']}"
            mcur.execute("""
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
            """, (s["student_id"], s["name"], s["department"], s["section"], s["register_number"], cls_name))
            print(f"Synced student to smartclass.sqlite: {s['name']} ({s['student_id']})")

        mconn.commit()
        mcur.execute("SELECT student_id, student_name, register_no FROM students;")
        m_students = mcur.fetchall()
        print("Final students in smartclass.sqlite:", m_students)
        mconn.close()

    print("\nDatabase cleanup and synchronization complete!")

if __name__ == "__main__":
    clean_and_sync()
