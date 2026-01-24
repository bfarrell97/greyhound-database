import sqlite3
conn = sqlite3.connect('greyhound_racing.db')
r = conn.execute("SELECT MIN(MeetingDate), MAX(MeetingDate) FROM RaceMeetings").fetchone()
print('Date range:', r)

r2 = conn.execute("""
    SELECT strftime('%Y', MeetingDate) as year, COUNT(*) as cnt
    FROM RaceMeetings 
    GROUP BY year 
    ORDER BY year
""").fetchall()
print('\nMeetings by year:')
for row in r2:
    print(f'  {row[0]}: {row[1]:,}')
    
r3 = conn.execute("""
    SELECT strftime('%Y', rm.MeetingDate) as year, COUNT(*) as cnt
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    GROUP BY year
    ORDER BY year
""").fetchall()
print('\nEntries by year:')
for row in r3:
    print(f'  {row[0]}: {row[1]:,}')
conn.close()
