import sqlite3

def check_schema():
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    with open('schema_info.txt', 'w') as f:
        tables = ['Greyhounds', 'GreyhoundEntries', 'Races', 'RaceMeetings']
        for t in tables:
            cursor.execute(f"PRAGMA table_info({t})")
            cols = [r[1] for r in cursor.fetchall()]
            f.write(f"\n{t} Columns:\n")
            f.write(str(cols) + "\n")
        
    conn.close()

if __name__ == "__main__":
    check_schema()
