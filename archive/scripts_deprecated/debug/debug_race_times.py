import sqlite3
import pandas as pd

def debug_race_times():
    conn = sqlite3.connect('greyhound_racing.db')
    
    print("--- BENDIGO RACE TIMES (DB) ---")
    query = """
    SELECT 
        r.RaceNumber,
        r.RaceTime,
        r.RaceName
    FROM Races r 
    JOIN RaceMeetings rm ON r.MeetingID=rm.MeetingID 
    JOIN Tracks t ON rm.TrackID=t.TrackID
    WHERE rm.MeetingDate='2025-12-13' AND t.TrackName='Bendigo'
    ORDER BY r.RaceNumber ASC
    """
    df = pd.read_sql_query(query, conn)
    print(df)
    conn.close()

if __name__ == "__main__":
    debug_race_times()
