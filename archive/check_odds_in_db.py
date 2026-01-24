"""Check if odds are in database"""
import sqlite3

conn = sqlite3.connect('greyhound_racing.db')
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM UpcomingBettingRunners WHERE CurrentOdds IS NOT NULL AND CurrentOdds > 0')
print(f'Runners with odds in DB: {cursor.fetchone()[0]}')

cursor.execute('SELECT GreyhoundName, BoxNumber, CurrentOdds FROM UpcomingBettingRunners WHERE CurrentOdds IS NOT NULL LIMIT 15')
print('\nSample runners WITH odds in database:')
for row in cursor.fetchall():
    print(f'{row[0]:<30} Box {row[1]}  ${row[2]:.2f}')

cursor.execute('SELECT GreyhoundName, BoxNumber, CurrentOdds FROM UpcomingBettingRunners WHERE CurrentOdds IS NULL LIMIT 15')
print('\nSample runners WITHOUT odds in database:')
for row in cursor.fetchall():
    print(f'{row[0]:<30} Box {row[1]}  {row[2]}')

conn.close()
