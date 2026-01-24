"""
Recreate UpcomingBetting tables after database restore
"""
import sqlite3

conn = sqlite3.connect('greyhound_racing.db')
c = conn.cursor()

# Create UpcomingBettingRaces table
c.execute('''
CREATE TABLE IF NOT EXISTS UpcomingBettingRaces (
    UpcomingBettingRaceID INTEGER PRIMARY KEY AUTOINCREMENT,
    MeetingDate TEXT,
    TrackName TEXT,
    TrackCode TEXT,
    RaceNumber INTEGER,
    RaceTime TEXT,
    Distance INTEGER,
    Grade TEXT,
    PrizeMoney REAL,
    UNIQUE(MeetingDate, TrackName, RaceNumber)
)
''')

# Create UpcomingBettingRunners table
c.execute('''
CREATE TABLE IF NOT EXISTS UpcomingBettingRunners (
    UpcomingBettingRunnerID INTEGER PRIMARY KEY AUTOINCREMENT,
    UpcomingBettingRaceID INTEGER,
    GreyhoundName TEXT,
    BoxNumber INTEGER,
    TrainerName TEXT,
    Form TEXT,
    BestTime REAL,
    Weight REAL,
    CurrentOdds REAL,
    FOREIGN KEY (UpcomingBettingRaceID) REFERENCES UpcomingBettingRaces(UpcomingBettingRaceID)
)
''')

conn.commit()
print('SUCCESS: UpcomingBettingRaces table created')
print('SUCCESS: UpcomingBettingRunners table created')

# Verify tables exist
c.execute('SELECT name FROM sqlite_master WHERE type="table" AND name LIKE "Upcoming%" ORDER BY name')
tables = c.fetchall()
print('\nUpcoming tables in database:')
for t in tables:
    c.execute(f'SELECT COUNT(*) FROM {t[0]}')
    count = c.fetchone()[0]
    print(f'  {t[0]}: {count} records')

conn.close()
print('\nDone! Tables are ready for upcoming race data.')
