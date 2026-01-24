import sqlite3
import pandas as pd
from datetime import datetime, timedelta

try:
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    # Check entries in the last 30 days
    last_month = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    print(f"Checking PrizeMoney since {last_month}")
    
    query = f'''
    SELECT 
        COUNT(*) as TotalEntries,
        SUM(CASE WHEN ge.CareerPrizeMoney IS NOT NULL AND ge.CareerPrizeMoney > 0 THEN 1 ELSE 0 END) as WithCareerMoney,
        SUM(CASE WHEN ge.PrizeMoney IS NOT NULL AND ge.PrizeMoney > 0 THEN 1 ELSE 0 END) as WithRaceMoney
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE rm.MeetingDate >= '{last_month}'
    '''
    
    row = cursor.execute(query).fetchone()
    total = row[0]
    with_career = row[1]
    with_race = row[2]
    
    print(f"Total Entries (last 30 days): {total}")
    print(f"Entries with CareerPrizeMoney: {with_career} ({(with_career/total)*100 if total else 0:.1f}%)")
    print(f"Entries with PrizeMoney: {with_race} ({(with_race/total)*100 if total else 0:.1f}%)")
        
    conn.close()
except Exception as e:
    print(f"Error: {e}")
