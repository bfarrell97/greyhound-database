import sqlite3
import pandas as pd

def check_patrick():
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT 
        ge.EntryID, g.GreyhoundName, ge.Box, r.RaceTime, r.RaceNumber, t.TrackName
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE (g.GreyhoundName LIKE '%PATRICK%')
    AND rm.MeetingDate = '2025-12-26'
    """
    df = pd.read_sql_query(query, conn)
    print("--- DB ENTRIES FOR DR PATRICK (2025-12-26) ---")
    print(df.to_string(index=False))
    conn.close()

if __name__ == "__main__":
    check_patrick()
