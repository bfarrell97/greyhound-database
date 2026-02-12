import sqlite3
import pandas as pd

DB_PATH = 'greyhound_racing.db'

def check():
    conn = sqlite3.connect(DB_PATH)
    try:
        query = """
        SELECT 
            rm.MeetingDate,
            r.RaceNumber,
            t.TrackName,
            ge.Position,
            ge.FinishTime,
            ge.Split,
            ge.FirstSplitPosition
        FROM GreyhoundEntries ge 
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID 
        JOIN Races r ON ge.RaceID = r.RaceID 
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID 
        JOIN Tracks t ON rm.TrackID = t.TrackID 
        WHERE g.GreyhoundName LIKE '%HOLLY ROSE%'
        ORDER BY rm.MeetingDate DESC
        LIMIT 10
        """
        df = pd.read_sql_query(query, conn)
        print(df)
    except Exception as e:
        print(e)
    finally:
        conn.close()

if __name__ == "__main__":
    check()
