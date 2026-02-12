import sqlite3
import pandas as pd

def check():
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT 
        strftime('%Y', rm.MeetingDate) as Year, 
        COUNT(*) as TotalRuns, 
        COUNT(ge.TopazSplit1) as TopazRuns,
        CAST(COUNT(ge.TopazSplit1) AS FLOAT) / COUNT(*) * 100 as Percentage
    FROM GreyhoundEntries ge 
    JOIN Races r ON ge.RaceID = r.RaceID 
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID 
    GROUP BY Year
    ORDER BY Year
    """
    df = pd.read_sql_query(query, conn)
    print(df)
    conn.close()

if __name__ == "__main__":
    try:
        check()
    except Exception as e:
        print(f"Error: {e}")
