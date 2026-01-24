
import sqlite3

def check_schema():
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    tables = ['GreyhoundEntries', 'Races', 'Tracks', 'RaceMeetings', 'Greyhounds']
    
    for table in tables:
        print(f"\n--- {table} ---")
        try:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            for col in columns:
                print(f"{col[1]} ({col[2]})")
        except Exception as e:
            print(f"Error: {e}")

    conn.close()

if __name__ == "__main__":
    check_schema()
