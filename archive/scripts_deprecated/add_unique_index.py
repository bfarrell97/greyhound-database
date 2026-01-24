import sqlite3

try:
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    print("Attempting to create UNIQUE INDEX on GreyhoundEntries(RaceID, GreyhoundID)...")
    
    cursor.execute('''
    CREATE UNIQUE INDEX IF NOT EXISTS idx_entries_race_dog 
    ON GreyhoundEntries(RaceID, GreyhoundID)
    ''')
    
    print("Index created successfully.")
    conn.commit()
    conn.close()
except Exception as e:
    print(f"Error: {e}")
