import os
import shutil
import sqlite3
import json
import faiss
from loguru import logger

def reset_enrollment_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Starting complete student enrollment data reset...")

    # 1. Reset NEW Enrollment Database (database/new_enrollment.sqlite)
    new_db_path = os.path.join(base_dir, "database", "new_enrollment.sqlite")
    if os.path.exists(new_db_path):
        conn = sqlite3.connect(new_db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM enrolled_embeddings;")
        cur.execute("DELETE FROM enrolled_students;")
        cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('enrolled_embeddings', 'enrolled_students');")
        conn.commit()
        conn.execute("VACUUM;")
        conn.close()
        print(f"[OK] Reset {new_db_path}: enrolled_students=0, enrolled_embeddings=0")

    # 2. Reset General Database (database/smartclass.sqlite)
    main_db_path = os.path.join(base_dir, "database", "smartclass.sqlite")
    if os.path.exists(main_db_path):
        conn = sqlite3.connect(main_db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM embeddings;")
        cur.execute("DELETE FROM students;")
        cur.execute("DELETE FROM attendance;")
        cur.execute("DELETE FROM enrollment_sources;")
        cur.execute("DELETE FROM enrollment_events;")
        cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('embeddings', 'attendance', 'enrollment_sources', 'enrollment_events');")
        conn.commit()
        conn.execute("VACUUM;")
        conn.close()
        print(f"[OK] Reset {main_db_path}: students=0, embeddings=0, attendance=0")

    # 3. Reset data/smartclass.sqlite if present
    data_db_path = os.path.join(base_dir, "data", "smartclass.sqlite")
    if os.path.exists(data_db_path):
        conn = sqlite3.connect(data_db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM embeddings;")
        cur.execute("DELETE FROM students;")
        cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('embeddings');")
        conn.commit()
        conn.execute("VACUUM;")
        conn.close()
        print(f"[OK] Reset {data_db_path}: students=0, embeddings=0")

    # 4. Reset FAISS Vector Index (database/faiss_index.bin and .meta.json)
    faiss_bin = os.path.join(base_dir, "database", "faiss_index.bin")
    faiss_meta = os.path.join(base_dir, "database", "faiss_index.bin.meta.json")

    sub_index = faiss.IndexFlatIP(512)
    empty_index = faiss.IndexIDMap2(sub_index)
    faiss.write_index(empty_index, faiss_bin)
    with open(faiss_meta, "w", encoding="utf-8") as f:
        json.dump({}, f, indent=2)
    print(f"[OK] Reset FAISS: total_vectors={empty_index.ntotal}, mappings=0")

    # 5. Clear stored enrollment photos (data/enrollment/*)
    enroll_dir = os.path.join(base_dir, "data", "enrollment")
    if os.path.exists(enroll_dir):
        for item in os.listdir(enroll_dir):
            item_path = os.path.join(enroll_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
            elif item.endswith((".jpg", ".png", ".jpeg")):
                try:
                    os.remove(item_path)
                except Exception:
                    pass
        print(f"[OK] Cleared {enroll_dir}: remaining student folders=0")

    # Remove temporary enrollment report if present
    report_file = os.path.join(base_dir, "data", "phase10_real_enrollment_report.json")
    if os.path.exists(report_file):
        try:
            os.remove(report_file)
            print(f"[OK] Removed {report_file}")
        except Exception:
            pass

    print("\nEnrollment Data Reset COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    reset_enrollment_data()
