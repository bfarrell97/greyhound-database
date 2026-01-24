import sqlite3
import pandas as pd
from datetime import datetime, timedelta

try:
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    # Check entries in the last 30 days
    last_month = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    print(f"Checking entries since {last_month}")
    
    query = f'''
    SELECT 
        COUNT(*) as TotalEntries,
        SUM(CASE WHEN FirstSplitPosition IS NOT NULL AND FirstSplitPosition != '' THEN 1 ELSE 0 END) as WithSplits,
        SUM(CASE WHEN FinishTimeBenchmarkLengths IS NOT NULL THEN 1 ELSE 0 END) as WithPace
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE rm.MeetingDate >= '{last_month}'
    '''
    
    row = cursor.execute(query).fetchone()
    total = row[0]
    with_splits = row[1]
    with_pace = row[2]
    
    print(f"Total Entries (last 30 days): {total}")
    print(f"Entries with Splits: {with_splits} ({(with_splits/total)*100:.1f}%)")
    print(f"Entries with Pace: {with_pace} ({(with_pace/total)*100:.1f}%)")
        
    conn.close()
except Exception as e:
    print(f"Error: {e}")
