import sqlite3
import os

db_files = [f for f in os.listdir('.') if f.endswith('.db')]
print('Available databases:', db_files)

if db_files:
    conn = sqlite3.connect(db_files[0])
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print('\nTables:', [t[0] for t in tables])
    
    cursor.execute('PRAGMA table_info(Races)')
    cols = cursor.fetchall()
    print('\nRaces columns:', [c[1] for c in cols])
    
    cursor.execute('PRAGMA table_info(RaceEntries)')
    cols = cursor.fetchall()
    print('\nRaceEntries columns:', [c[1] for c in cols])
    
    # Check if Rating exists and has data
    try:
        cursor.execute('SELECT COUNT(*), COUNT(Rating), COUNT(DISTINCT Rating) FROM RaceEntries')
        result = cursor.fetchone()
        print(f'\nRating column stats: Total rows={result[0]}, Non-null ratings={result[1]}, Distinct values={result[2]}')
    except Exception as e:
        print(f'\nRating column error: {e}')
