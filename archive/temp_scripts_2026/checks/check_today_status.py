import sqlite3
import pandas as pd
from datetime import datetime

# Connect to DB
conn = sqlite3.connect('greyhound_racing.db')
today_str = datetime.now().strftime('%Y-%m-%d')
print(f"Checking races for: {today_str}")

# Query today's races
query = f"""
SELECT 
    t.TrackName, r.RaceNumber, r.RaceTime, 
    COUNT(ge.EntryID) as Runners,
    COUNT(ge.Price5Min) as Price_Count,
    COUNT(ge.BSP) as Finished_Count
FROM Races r
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
JOIN GreyhoundEntries ge ON ge.RaceID = r.RaceID
WHERE rm.MeetingDate = '{today_str}'
GROUP BY t.TrackName, r.RaceNumber
ORDER BY r.RaceTime
"""

df = pd.read_sql_query(query, conn)
conn.close()

if df.empty:
    print("NO RACES found in DB for today.")
else:
    print(f"Found {len(df)} races for today.")
    print("Sample of upcoming/recent races:")
    print(df.head(20))
    
    print("\nSummary:")
    print(f"Total Runners: {df['Runners'].sum()}")
    print(f"Runners with Price5Min (Captured): {df['Price_Count'].sum()}")
    print(f"Runners Finished (BSP): {df['Finished_Count'].sum()}")
