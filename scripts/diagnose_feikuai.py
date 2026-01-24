import pandas as pd
import sqlite3

def diagnose():
    conn = sqlite3.connect('greyhound_racing.db')
    
    # 1. Check Today's Entry for FEIKUAI CALVIN
    print("--- TODAY'S ENTRY ---")
    query = """
    SELECT 
        ge.EntryID, ge.GreyhoundID, ge.TrainerID, 
        r.Distance, r.RaceNumber, r.RaceTime,
        rm.MeetingDate, t.TrackName
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE g.GreyhoundName LIKE '%FEIKUAI CALVIN%'
    AND rm.MeetingDate = '2025-12-26'
    """
    df_today = pd.read_sql_query(query, conn)
    print(df_today.to_string(index=False))
    
    if df_today.empty:
        print("No entry found for today!")
    
    # 2. Check History (Last 5 races)
    print("\n--- HISTORY (Last 5) ---")
    query_hist = """
    SELECT 
        rm.MeetingDate, t.TrackName, r.Distance,
        ge.Position, ge.Split, ge.FirstSplitPosition, ge.TopazSplit1
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE g.GreyhoundName LIKE '%FEIKUAI CALVIN%'
    ORDER BY rm.MeetingDate DESC
    LIMIT 6
    """
    df_hist = pd.read_sql_query(query_hist, conn)
    print(df_hist.to_string(index=False))
    
    conn.close()

if __name__ == "__main__":
    diagnose()
