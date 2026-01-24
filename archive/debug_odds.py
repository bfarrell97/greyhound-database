"""Debug the track data"""
import sqlite3
import pandas as pd

DB_PATH = 'greyhound_racing.db'

conn = sqlite3.connect(DB_PATH)

# Check Nowra data
query = """
SELECT COUNT(*) as total, 
       SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
       COUNT(*) FILTER (WHERE ge.StartingPrice >= '1.5' AND ge.StartingPrice < '2.0') as in_range,
       SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) 
          FILTER (WHERE ge.StartingPrice >= '1.5' AND ge.StartingPrice < '2.0') as wins_in_range
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE t.TrackName = 'Nowra'
  AND rm.MeetingDate >= '2025-01-01'
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
"""

result = pd.read_sql_query(query, conn)
print("Nowra all odds:")
print(result)

# Check if StartingPrice is stored as text
query2 = """
SELECT StartingPrice, COUNT(*) 
FROM GreyhoundEntries 
WHERE StartingPrice IS NOT NULL 
LIMIT 10
"""
samples = pd.read_sql_query(query2, conn)
print("\nStartingPrice samples:")
print(samples)

conn.close()
