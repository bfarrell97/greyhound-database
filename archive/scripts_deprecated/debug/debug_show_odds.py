import sqlite3
import pandas as pd

def debug_show_odds():
    conn = sqlite3.connect('greyhound_racing.db')
    
    # query breakdown by track
    print("--- ODDS COVERAGE BY TRACK ---")
    query_tracks = """
    SELECT 
        t.TrackName,
        COUNT(ge.EntryID) as TotalEntries,
        COUNT(ge.StartingPrice) as WithOdds,
        ROUND(CAST(COUNT(ge.StartingPrice) as FLOAT) / COUNT(ge.EntryID) * 100, 1) as CoveragePct
    FROM GreyhoundEntries ge 
    JOIN Races r ON ge.RaceID=r.RaceID 
    JOIN RaceMeetings rm ON r.MeetingID=rm.MeetingID 
    JOIN Tracks t ON rm.TrackID=t.TrackID
    WHERE rm.MeetingDate='2025-12-13'
    GROUP BY t.TrackName
    ORDER BY CoveragePct DESC
    """
    df_tracks = pd.read_sql_query(query_tracks, conn)
    print(df_tracks)
    
    print("\n--- SAMPLE WITH ODDS ---")
    query_with = """
    SELECT r.RaceNumber, t.TrackName, ge.Box, g.GreyhoundName, ge.StartingPrice
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID=r.RaceID 
    JOIN RaceMeetings rm ON r.MeetingID=rm.MeetingID 
    JOIN Tracks t ON rm.TrackID=t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID=g.GreyhoundID
    WHERE rm.MeetingDate='2025-12-13' AND ge.StartingPrice IS NOT NULL
    LIMIT 10
    """
    df_with = pd.read_sql_query(query_with, conn)
    print(df_with)
    
    print("\n--- SAMPLE WITHOUT ODDS ---")
    query_without = """
    SELECT r.RaceNumber, t.TrackName, ge.Box, g.GreyhoundName
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID=r.RaceID 
    JOIN RaceMeetings rm ON r.MeetingID=rm.MeetingID 
    JOIN Tracks t ON rm.TrackID=t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID=g.GreyhoundID
    WHERE rm.MeetingDate='2025-12-13' AND ge.StartingPrice IS NULL
    LIMIT 10
    """
    df_without = pd.read_sql_query(query_without, conn)
    print(df_without)
    
    conn.close()

if __name__ == "__main__":
    debug_show_odds()
