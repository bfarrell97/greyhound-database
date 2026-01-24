import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    # Check Split Coverage
    print(f"Checking Split vs FirstSplitPosition coverage")
    
    query = f'''
    SELECT 
        COUNT(*) as Total,
        SUM(CASE WHEN Split IS NOT NULL THEN 1 ELSE 0 END) as WithSplit,
        SUM(CASE WHEN FirstSplitPosition IS NOT NULL THEN 1 ELSE 0 END) as WithPos
    FROM GreyhoundEntries
    '''
    
    row = cursor.execute(query).fetchone()
    print(f"Total: {row[0]}")
    print(f"With 'Split' (Time): {row[1]}")
    print(f"With 'FirstSplitPosition': {row[2]}")
        
    conn.close()
except Exception as e:
    print(f"Error: {e}")
