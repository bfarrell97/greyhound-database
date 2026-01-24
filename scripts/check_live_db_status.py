
import sqlite3
import pandas as pd

conn = sqlite3.connect('greyhound_racing.db')
pd.set_option('display.max_colwidth', None)
query = "SELECT SelectionName, Price, Status, Result, PlacedDate FROM LiveBets WHERE SelectionName LIKE '%SPRING SABALENKA%' ORDER BY PlacedDate DESC"
df = pd.read_sql_query(query, conn)
print(f"Rows for SPRING SABALENKA: {len(df)}")
if len(df) > 0:
    print(df.to_string())
else:
    print("No SPRING SABALENKA found.")
conn.close()
