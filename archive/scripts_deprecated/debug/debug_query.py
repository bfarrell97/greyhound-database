import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('greyhound_racing.db')
    
    # query from app.py
    hist_query = '''
    SELECT 
        g.GreyhoundID,
        g.GreyhoundName,
        AVG(ge.FirstSplitPosition) as HistAvgSplit,
        COUNT(*) as RaceCount,
        MAX(ge.CareerPrizeMoney) as CareerPrizeMoney
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE ge.FirstSplitPosition IS NOT NULL 
      AND ge.FirstSplitPosition != ''
      AND ge.Position IS NOT NULL
    GROUP BY ge.GreyhoundID
    LIMIT 10
    '''
    
    print("Running full query...")
    df = pd.read_sql_query(hist_query, conn)
    print(f"Result count: {len(df)}")
    print(df.to_string())

    # Check join integrity
    print("\nChecking Joins:")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM GreyhoundEntries")
    print(f"Entries: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM GreyhoundEntries ge JOIN Races r ON ge.RaceID = r.RaceID")
    print(f"Entries + Races: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM GreyhoundEntries ge JOIN Races r ON ge.RaceID = r.RaceID JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID")
    print(f"Entries + Races + Meetings: {cursor.fetchone()[0]}")
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")
