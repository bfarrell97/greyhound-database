import sqlite3
import pandas as pd

def check_duplicate_races():
    conn = sqlite3.connect('greyhound_racing.db')
    
    print("--- RACES FOR BENDIGO TODAY ---")
    query = """
    SELECT 
        r.RaceID,
        r.RaceNumber,
        r.RaceName,
        t.TrackName,
        r.RaceTime,
        (SELECT COUNT(*) FROM GreyhoundEntries ge WHERE ge.RaceID=r.RaceID) as Entries,
        (SELECT COUNT(*) FROM GreyhoundEntries ge WHERE ge.RaceID=r.RaceID AND ge.StartingPrice IS NOT NULL) as WithOdds
    FROM Races r 
    JOIN RaceMeetings rm ON r.MeetingID=rm.MeetingID 
    JOIN Tracks t ON rm.TrackID=t.TrackID
    WHERE rm.MeetingDate='2025-12-13' AND t.TrackName LIKE '%Bendigo%'
    ORDER BY r.RaceNumber, r.RaceID
    """
    df = pd.read_sql_query(query, conn)
    print(df)
    conn.close()

if __name__ == "__main__":
    check_duplicate_races()
