"""Find dates with data for both DB and BSP for debugging"""
import sqlite3
c = sqlite3.connect('greyhound_racing.db')

# Bet Deluxe Capalaba dates needing BSP
query = """
SELECT rm.MeetingDate, COUNT(*) 
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE t.TrackName = 'Bet Deluxe Capalaba' AND ge.BSP IS NULL
AND rm.MeetingDate BETWEEN '2025-09-01' AND '2025-11-30'
GROUP BY rm.MeetingDate
ORDER BY COUNT(*) DESC LIMIT 10
"""
print('Bet Deluxe Capalaba dates needing BSP:')
for row in c.execute(query).fetchall():
    print(f'  {row[0]}: {row[1]} entries')
