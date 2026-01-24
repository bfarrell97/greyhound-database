import sqlite3
import pandas as pd

conn = sqlite3.connect('greyhound_racing.db')
query = "SELECT Min(MeetingDate), Max(MeetingDate) FROM RaceMeetings"
df = pd.read_sql_query(query, conn)
print(df)
conn.close()
