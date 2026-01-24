import sqlite3
import pandas as pd
from datetime import datetime

db_path = 'greyhound_racing.db'
conn = sqlite3.connect(db_path)

today = datetime.now().strftime('%Y-%m-%d')
print(f"Checking RaceTime for {today}")

query = """
SELECT RaceID, RaceNumber, RaceTime FROM Races
WHERE MeetingID IN (SELECT MeetingID FROM RaceMeetings WHERE MeetingDate = ?)
"""
df = pd.read_sql_query(query, conn, params=(today,))
print(df.head(20))
conn.close()
