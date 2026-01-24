"""Diagnose why V42 produces fewer signals in backtest vs training"""
import sqlite3
import pandas as pd

conn = sqlite3.connect('greyhound_racing.db')

# 1. Check Price5Min data availability
print("=== Price5Min Data Availability ===")
query = """
SELECT 
    strftime('%Y', rm.MeetingDate) as Year,
    COUNT(*) as TotalRows,
    SUM(CASE WHEN ge.Price5Min > 0 THEN 1 ELSE 0 END) as WithPrice5Min
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
GROUP BY strftime('%Y', rm.MeetingDate)
ORDER BY Year
"""
df = pd.read_sql_query(query, conn)
print(df.to_string(index=False))

# 2. Check 2025 specifically (training data period)
print("\n=== 2025 Data Details ===")
query2 = """
SELECT COUNT(*) as cnt FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate >= '2025-01-01'
AND ge.Price5Min > 0
AND ge.BSP > 0
"""
df2 = pd.read_sql_query(query2, conn)
print(f"Rows with both Price5Min and BSP in 2025: {df2.iloc[0]['cnt']}")

# 3. Check 2024 specifically
print("\n=== 2024 Data Details ===")
query3 = """
SELECT COUNT(*) as cnt FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate >= '2024-01-01' AND rm.MeetingDate < '2025-01-01'
AND ge.Price5Min > 0
"""
df3 = pd.read_sql_query(query3, conn)
print(f"Rows with Price5Min in 2024: {df3.iloc[0]['cnt']}")

conn.close()

print("\n=== Training vs Backtest Comparison ===")
print("Training used: 2025 data only")
print("Backtest used: 2024-2025 data")
print("If 2024 has very few Price5Min values, that explains the discrepancy.")
