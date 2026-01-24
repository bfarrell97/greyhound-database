"""Check if specific dogs from GUI are in database"""
import sqlite3

conn = sqlite3.connect('greyhound_racing.db')
c = conn.cursor()

dogs = ['STRAYBOUND', 'ORSON CHARLES', 'WAR MAN', 'MISS TWISTER',
        'FROM NOW ON', 'SPRING ZEAL', 'SAINT OLLIE']

print('\n=== Checking dogs in predictions ===')
for dog in dogs:
    c.execute('''
        SELECT ubr.TrackName, ubr.RaceNumber, ur.BoxNumber, ur.CurrentOdds
        FROM UpcomingBettingRunners ur
        JOIN UpcomingBettingRaces ubr ON ur.UpcomingBettingRaceID = ubr.UpcomingBettingRaceID
        WHERE ur.GreyhoundName = ? AND ubr.MeetingDate = "2025-12-08"
    ''', (dog,))

    rows = c.fetchall()
    if rows:
        print(f'{dog:<20} - {len(rows)} entries')
        for row in rows:
            print(f'  {row[0]:<20} R{row[1]}  Box {row[2]}  Odds: {row[3]}')
    else:
        print(f'{dog:<20} - NOT IN DATABASE')

conn.close()
