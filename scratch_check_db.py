import sqlite3

def check_db(name, path):
    print(f"=== {name} ({path}) ===")
    try:
        conn = sqlite3.connect(path)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in c.fetchall()]
        print("Tables:", tables)
        for t in tables:
            c.execute(f"SELECT count(*) FROM {t}")
            count = c.fetchone()[0]
            print(f"Table '{t}' has {count} rows")
            c.execute(f"SELECT * FROM {t} LIMIT 10")
            cols = [d[0] for d in c.description]
            print(f"  Columns: {cols}")
            for row in c.fetchall():
                row_disp = [str(x)[:30] if isinstance(x, (bytes, str)) else x for x in row]
                print(f"  Row: {row_disp}")
        conn.close()
    except Exception as e:
        print(f"Error checking {name}: {e}")

if __name__ == "__main__":
    check_db("NEW ENROLLMENT", "database/new_enrollment.sqlite")
    print()
    check_db("SMARTCLASS", "database/smartclass.sqlite")
