import sqlite3
import pandas as pd

def debug_bendigo_v2():
    conn = sqlite3.connect('greyhound_racing.db')
    
    print("--- BENDIGO ODDS (Current State) ---")
    query = """
    SELECT 
        r.RaceNumber,
        COUNT(ge.EntryID) as Total,
        SUM(CASE WHEN ge.StartingPrice IS NOT NULL THEN 1 ELSE 0 END) as WithOdds,
        GROUP_CONCAT(CASE WHEN ge.Box=1 THEN ge.StartingPrice ELSE NULL END) as Box1_Price
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
    conn.close()

if __name__ == "__main__":
    debug_bendigo_v2()
