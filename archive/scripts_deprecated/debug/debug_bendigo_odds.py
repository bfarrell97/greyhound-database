import sqlite3
import pandas as pd

def debug_bendigo():
    conn = sqlite3.connect('greyhound_racing.db')
    
    print("--- BENDIGO ODDS BREAKDOWN (Today) ---")
    query = """
    SELECT 
        r.RaceNumber,
        r.RaceTime,
        COUNT(ge.EntryID) as Runners,
        SUM(CASE WHEN ge.StartingPrice IS NOT NULL THEN 1 ELSE 0 END) as WithOdds,
        CASE WHEN SUM(CASE WHEN ge.StartingPrice IS NOT NULL THEN 1 ELSE 0 END) = 0 THEN 'CLOSED/PAST' ELSE 'OPEN' END as Status
    FROM GreyhoundEntries ge 
    JOIN Races r ON ge.RaceID=r.RaceID 
    JOIN RaceMeetings rm ON r.MeetingID=rm.MeetingID 
    JOIN Tracks t ON rm.TrackID=t.TrackID
    WHERE rm.MeetingDate='2025-12-13' AND t.TrackName='Bendigo'
    GROUP BY r.RaceNumber
    ORDER BY r.RaceNumber ASC
    """
    df = pd.read_sql_query(query, conn)
    print(df)
    
    print("\n--- SAMPLE ODDS (Race 10) ---")
    query_sample = """
    SELECT ge.Box, g.GreyhoundName, ge.StartingPrice
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID=r.RaceID 
    JOIN RaceMeetings rm ON r.MeetingID=rm.MeetingID 
    JOIN Tracks t ON rm.TrackID=t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID=g.GreyhoundID
    WHERE rm.MeetingDate='2025-12-13' AND t.TrackName='Bendigo' AND r.RaceNumber=10
    ORDER BY ge.Box ASC
    """
    df_sample = pd.read_sql_query(query_sample, conn)
    print(df_sample)
    
    conn.close()

if __name__ == "__main__":
    debug_bendigo()
