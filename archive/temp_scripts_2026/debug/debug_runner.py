import sqlite3
import pandas as pd

conn = sqlite3.connect('greyhound_racing.db')
query = """
SELECT 
    g.GreyhoundName, ge.Box, ge.Price5Min, ge.StartingPrice
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE t.TrackName = 'Capalaba' 
AND rm.MeetingDate = '2025-12-31'
AND r.RaceNumber = 10
ORDER BY ge.Box
"""
df = pd.read_sql_query(query, conn)
print(df.to_string())
conn.close()
