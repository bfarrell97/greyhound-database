import sqlite3
c = sqlite3.connect('greyhound_racing.db')
r = c.execute("SELECT COUNT(*) FROM GreyhoundEntries ge JOIN Races r ON ge.RaceID = r.RaceID JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID WHERE rm.MeetingDate BETWEEN '2025-11-01' AND '2025-11-30'").fetchone()
print(f'November 2025 entries: {r[0]:,}')
