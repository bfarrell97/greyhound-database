import sqlite3
import pandas as pd

def verify():
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT 
        rm.MeetingDate, 
        Count(*) as TotalEntries, 
        Count(ge.TopazSplit1) as HasSplit1, 
        Count(ge.TopazPIR) as HasPIR,
        Count(ge.TopazComment) as HasComment
    FROM GreyhoundEntries ge 
    JOIN Races r ON ge.RaceID = r.RaceID 
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID 
    WHERE rm.MeetingDate >= '2024-12-01' 
    GROUP BY rm.MeetingDate 
    ORDER BY rm.MeetingDate DESC 
    LIMIT 10
    """
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))
    
    print("\nChecking populated date range:")
    range_query = """
    SELECT MIN(rm.MeetingDate), MAX(rm.MeetingDate), COUNT(*)
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE ge.TopazSplit1 IS NOT NULL
    """
    range_df = pd.read_sql_query(range_query, conn)
    print(range_df.to_string(index=False))

    print("\nSample Data (Latest):")
    sample_query = """
    SELECT ge.TopazSplit1, ge.TopazPIR, ge.TopazComment, ge.FinishTime 
    FROM GreyhoundEntries ge 
    WHERE ge.TopazSplit1 IS NOT NULL 
    ORDER BY ge.EntryID DESC LIMIT 5
    """
    sample = pd.read_sql_query(sample_query, conn)
    print(sample.to_string(index=False))
    conn.close()

if __name__ == "__main__":
    verify()
