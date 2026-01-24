import sqlite3
import pandas as pd

conn = sqlite3.connect('greyhound_racing.db')

query = """
SELECT 
    COUNT(*) as total, 
    SUM(CASE WHEN BSP IS NOT NULL THEN 1 ELSE 0 END) as with_bsp, 
    SUM(CASE WHEN FirstSplitPosition IS NOT NULL AND FirstSplitPosition != '' THEN 1 ELSE 0 END) as with_split, 
    SUM(CASE WHEN FinishTimeBenchmarkLengths IS NOT NULL THEN 1 ELSE 0 END) as with_pace, 
    SUM(CASE WHEN BSP IS NOT NULL AND FirstSplitPosition IS NOT NULL AND FirstSplitPosition != '' THEN 1 ELSE 0 END) as bsp_and_split,
    SUM(CASE WHEN BSP IS NOT NULL AND FinishTimeBenchmarkLengths IS NOT NULL THEN 1 ELSE 0 END) as bsp_and_pace
FROM GreyhoundEntries
"""

df = pd.read_sql_query(query, conn)
print(df.to_string())
conn.close()
