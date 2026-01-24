import sqlite3

try:
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    cursor.execute('PRAGMA table_info(GreyhoundEntries)')
    columns = cursor.fetchall()
    print("Columns in GreyhoundEntries:")
    for col in columns:
        print(f"{col[1]} ({col[2]})")
    conn.close()
except Exception as e:
    print(f"Error: {e}")
