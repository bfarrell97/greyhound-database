"""
Add BSP Column to GreyhoundEntries
"""
import sqlite3
import os

DB_PATH = 'greyhound_racing.db'

def run():
    print(f"Adding BSP column to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("ALTER TABLE GreyhoundEntries ADD COLUMN BSP REAL")
        print("Success: Added BSP column.")
    except sqlite3.OperationalError as e:
        if "duplicate column" in str(e):
            print("Info: BSP column already exists.")
        else:
            print(f"Error: {e}")
            
    conn.commit()
    conn.close()

if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        run()
    else:
        print(f"Error: Database not found at {DB_PATH}")
