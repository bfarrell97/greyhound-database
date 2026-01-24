
import sqlite3
import pandas as pd

conn = sqlite3.connect('greyhound_racing.db')
query = "SELECT * FROM LiveBets WHERE SelectionName LIKE '%WONG%' AND MeetingDate = '2026-01-01'"
df = pd.read_sql_query(query, conn)
print(df.to_string())
conn.close()
