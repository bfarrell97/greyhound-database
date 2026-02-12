import sqlite3
import os

DB_PATH = 'greyhound_racing.db'

def add_columns():
    print(f"Adding Topaz columns to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    columns = [
        ('TopazSplit1', 'REAL'),
        ('TopazSplit2', 'REAL'),
        ('TopazPIR', 'TEXT'),
        ('TopazComment', 'TEXT')
    ]
    
    for col_name, col_type in columns:
        try:
            cursor.execute(f"ALTER TABLE GreyhoundEntries ADD COLUMN {col_name} {col_type}")
            print(f"Added column: {col_name}")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e):
                print(f"Column {col_name} already exists.")
            else:
                print(f"Error adding {col_name}: {e}")
                
    conn.commit()
    conn.close()
    print("Schema update complete.")

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
    else:
        add_columns()
