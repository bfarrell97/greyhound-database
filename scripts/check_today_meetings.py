
import sqlite3
from datetime import datetime
import pandas as pd

DB_PATH = "greyhound_db.sqlite"

def check_today():
    conn = sqlite3.connect(DB_PATH)
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"Checking for meetings on: {today}")
    
    query = f"""
    SELECT t.TrackName, count(*) as RaceCount 
    FROM RaceMeetings rm
    JOIN Races r ON rm.MeetingID = r.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate = '{today}'
    GROUP BY t.TrackName
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("NO MEETINGS FOUND FOR TODAY IN DB!")
    else:
        print(df)

if __name__ == "__main__":
    check_today()
