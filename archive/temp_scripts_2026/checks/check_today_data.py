import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "greyhound_racing.db"

def check_today():
    print("Checking LATEST data in DB...")
    
    conn = sqlite3.connect(DB_PATH)
    # Check latest Meeting Dates
    query_meetings = "SELECT MeetingDate, COUNT(*) as Count FROM RaceMeetings GROUP BY MeetingDate ORDER BY MeetingDate DESC LIMIT 10"
    
    # Check latest Entries with Price
    query_prices = """
    SELECT rm.MeetingDate, t.TrackName, r.RaceNumber, ge.Price5Min, ge.BSP
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.Price5Min IS NOT NULL
    ORDER BY rm.MeetingDate DESC, r.RaceTime DESC
    LIMIT 20
    """
    
    try:
        df_meetings = pd.read_sql_query(query_meetings, conn)
        print("\n--- Latest Meeting Dates ---")
        print(df_meetings.to_string(index=False))
        
        df_prices = pd.read_sql_query(query_prices, conn)
        print("\n--- Latest Captured Prices ---")
        print(df_prices.to_string(index=False))
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_today()
