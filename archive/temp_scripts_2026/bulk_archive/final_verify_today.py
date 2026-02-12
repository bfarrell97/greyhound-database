import sqlite3
import pandas as pd

def final_verify():
    conn = sqlite3.connect('greyhound_racing.db')
    
    print("--- TODAY'S RUNNER DATA STATUS (2025-12-26) ---")
    query = """
    SELECT 
        count(*) as total_runners,
        sum(case when ge.TrainerID IS NOT NULL then 1 else 0 end) as has_trainer,
        sum(case when r.Distance IS NOT NULL then 1 else 0 end) as has_distance,
        sum(case when ge.Split IS NOT NULL then 1 else 0 end) as has_today_split
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE rm.MeetingDate = '2025-12-26'
    """
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))
    
    print("\n--- SAMPLE RUNNERS (TODAY) ---")
    query_sample = """
    SELECT g.GreyhoundName, ge.TrainerID, r.Distance
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE rm.MeetingDate = '2025-12-26'
    LIMIT 10
    """
    df_sample = pd.read_sql_query(query_sample, conn)
    print(df_sample.to_string(index=False))
    
    conn.close()

if __name__ == "__main__":
    final_verify()
