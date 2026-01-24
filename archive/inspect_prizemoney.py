import sqlite3
import pandas as pd

conn = sqlite3.connect('greyhound_racing.db')
df = pd.read_sql_query("SELECT GreyhoundName, Prizemoney FROM Greyhounds ORDER BY RANDOM() LIMIT 20", conn)
print(df)
print("\nUnique non-null values sample:")
print(pd.read_sql_query("SELECT DISTINCT Prizemoney FROM Greyhounds WHERE Prizemoney IS NOT NULL LIMIT 20", conn))
conn.close()
