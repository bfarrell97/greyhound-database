"""Quick check of upcoming betting races data"""
import sqlite3

conn = sqlite3.connect('greyhound_racing.db')
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM UpcomingBettingRaces')
print(f'Races: {cursor.fetchone()[0]}')

cursor.execute('SELECT COUNT(*) FROM UpcomingBettingRunners')
print(f'Runners: {cursor.fetchone()[0]}')

cursor.execute('''
    SELECT r.MeetingDate, r.TrackName, r.RaceNumber, r.Distance, COUNT(ru.UpcomingBettingRunnerID) as runners
    FROM UpcomingBettingRaces r
    LEFT JOIN UpcomingBettingRunners ru ON r.UpcomingBettingRaceID = ru.UpcomingBettingRaceID
    GROUP BY r.UpcomingBettingRaceID
    ORDER BY r.TrackName, r.RaceNumber
    LIMIT 10
''')
print('\nSample races:')
for row in cursor.fetchall():
    print(f'  {row[0]} {row[1]} R{row[2]} {row[3]}m - {row[4]} runners')

cursor.execute('''
    SELECT r.TrackName, r.RaceNumber, ru.BoxNumber, ru.GreyhoundName, ru.CurrentOdds
    FROM UpcomingBettingRunners ru
    JOIN UpcomingBettingRaces r ON ru.UpcomingBettingRaceID = r.UpcomingBettingRaceID
    LIMIT 10
''')
print('\nSample runners:')
for row in cursor.fetchall():
    odds = row[4] if row[4] else "N/A"
    print(f'  {row[0]} R{row[1]} - Box {row[2]}: {row[3]} @ ${odds}')

conn.close()
