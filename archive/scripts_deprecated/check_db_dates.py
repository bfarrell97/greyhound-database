"""Check DB date coverage"""
import sqlite3
conn = sqlite3.connect('greyhound_racing.db')

print('DB date range:')
cursor = conn.execute('SELECT MIN(MeetingDate), MAX(MeetingDate) FROM RaceMeetings')
print(cursor.fetchone())

print('\nEntries by month for 2025:')
query = """
SELECT strftime('%Y-%m', rm.MeetingDate) as YM, COUNT(*)
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate >= '2025-01-01'
GROUP BY YM
ORDER BY YM
"""
for row in conn.execute(query).fetchall():
    print(f'  {row[0]}: {row[1]:,} entries')

print('\nEntries with BSP NULL by month for 2025:')
query = """
SELECT strftime('%Y-%m', rm.MeetingDate) as YM, COUNT(*)
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE ge.BSP IS NULL AND rm.MeetingDate >= '2025-01-01'
GROUP BY YM
ORDER BY YM
"""
for row in conn.execute(query).fetchall():
    print(f'  {row[0]}: {row[1]:,} entries need BSP')

conn.close()
