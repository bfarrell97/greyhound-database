import sqlite3
import pandas as pd

def dump_bendigo_detailed():
    conn = sqlite3.connect('greyhound_racing.db')
    
    print("--- BENDIGO R1 DETAILED ---")
    query = """
    SELECT 
        r.RaceNumber,
        g.GreyhoundName,
        ge.Box,
        ge.StartingPrice
    FROM GreyhoundEntries ge 
    JOIN Races r ON ge.RaceID=r.RaceID 
    JOIN RaceMeetings rm ON r.MeetingID=rm.MeetingID 
    JOIN Tracks t ON rm.TrackID=t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID=g.GreyhoundID
    WHERE rm.MeetingDate='2025-12-13' AND t.TrackName='Bendigo' AND r.RaceNumber=1
    """
    df = pd.read_sql_query(query, conn)
    print(df)
    conn.close()

if __name__ == "__main__":
    dump_bendigo_detailed()
