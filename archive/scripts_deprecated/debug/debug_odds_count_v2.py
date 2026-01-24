import sqlite3
import pandas as pd

conn = sqlite3.connect('greyhound_racing.db')
query = """
SELECT 
    COUNT(ge.EntryID) as TotalEntries, 
    COUNT(ge.StartingPrice) as WithOdds 
FROM GreyhoundEntries ge 
JOIN Races r ON ge.RaceID=r.RaceID 
JOIN RaceMeetings rm ON r.MeetingID=rm.MeetingID 
WHERE rm.MeetingDate='2025-12-13'
"""
df = pd.read_sql_query(query, conn)
print(df)

# detailed breakdown by track
query_tracks = """
SELECT 
    t.TrackName,
    COUNT(ge.EntryID) as Total,
    COUNT(ge.StartingPrice) as WithOdds
FROM GreyhoundEntries ge 
JOIN Races r ON ge.RaceID=r.RaceID 
JOIN RaceMeetings rm ON r.MeetingID=rm.MeetingID 
JOIN Tracks t ON rm.TrackID=t.TrackID
WHERE rm.MeetingDate='2025-12-13'
GROUP BY t.TrackName
"""
df_tracks = pd.read_sql_query(query_tracks, conn)
print("\nBy Track:")
print(df_tracks)

conn.close()
