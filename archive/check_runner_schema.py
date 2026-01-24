import sqlite3

conn = sqlite3.connect('greyhound_racing.db')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(UpcomingBettingRunners)")
cols = cursor.fetchall()
print("UpcomingBettingRunners columns:")
for col in cols:
    print(f"  {col}")
conn.close()
