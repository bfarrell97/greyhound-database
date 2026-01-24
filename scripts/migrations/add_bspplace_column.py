
import sqlite3
import os

DB_PATH = 'greyhound_racing.db'

def migrate():
    print(f"Connecting to {DB_PATH}...")
    if not os.path.exists(DB_PATH):
        print("Database not found!")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("Checking if BSPPlace column exists...")
    cursor.execute("PRAGMA table_info(GreyhoundEntries)")
    columns = [row[1] for row in cursor.fetchall()]

    if 'BSPPlace' not in columns:
        print("Adding BSPPlace column to GreyhoundEntries...")
        try:
            cursor.execute("ALTER TABLE GreyhoundEntries ADD COLUMN BSPPlace REAL")
            conn.commit()
            print("[OK] Column BSPPlace added successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to add column: {e}")
            conn.rollback()
    else:
        print("[SKIP] BSPPlace column already exists.")

    conn.close()

if __name__ == "__main__":
    migrate()
