import sqlite3
import pandas as pd

conn = sqlite3.connect('greyhound_racing.db')

# Check what dates have upcoming races
query = """
SELECT DISTINCT MeetingDate
FROM UpcomingBettingRaces
ORDER BY MeetingDate DESC
LIMIT 10
"""

dates = pd.read_sql_query(query, conn)
print("Available upcoming race dates:")
print(dates.to_string(index=False))

conn.close()
