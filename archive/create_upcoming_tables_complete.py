"""
Recreate UpcomingBetting tables with ALL required columns
"""
import sqlite3

conn = sqlite3.connect('greyhound_racing.db')
c = conn.cursor()

# Drop existing tables
c.execute('DROP TABLE IF EXISTS UpcomingBettingRunners')
c.execute('DROP TABLE IF EXISTS UpcomingBettingRaces')

# Create UpcomingBettingRaces table with ALL columns
c.execute('''
CREATE TABLE UpcomingBettingRaces (
    UpcomingBettingRaceID INTEGER PRIMARY KEY AUTOINCREMENT,
    MeetingDate TEXT,
    TrackCode TEXT,
    TrackName TEXT,
    RaceNumber INTEGER,
    RaceTime TEXT,
    Distance INTEGER,
    RaceType TEXT,
    Grade TEXT,
    PrizeMoney REAL,
    LastUpdated TEXT,
    UNIQUE(MeetingDate, TrackCode, RaceNumber)
)
''')

# Create UpcomingBettingRunners table
c.execute('''
CREATE TABLE UpcomingBettingRunners (
    UpcomingBettingRunnerID INTEGER PRIMARY KEY AUTOINCREMENT,
    UpcomingBettingRaceID INTEGER,
    GreyhoundName TEXT,
    BoxNumber INTEGER,
    TrainerName TEXT,
    Form TEXT,
    BestTime REAL,
    Weight REAL,
    CurrentOdds REAL,
    LastUpdated TEXT,
    FOREIGN KEY (UpcomingBettingRaceID) REFERENCES UpcomingBettingRaces(UpcomingBettingRaceID)
)
''')

conn.commit()
print('SUCCESS: UpcomingBettingRaces table created with ALL columns')
print('SUCCESS: UpcomingBettingRunners table created')

# Verify tables exist
c.execute('SELECT name FROM sqlite_master WHERE type="table" AND name LIKE "Upcoming%" ORDER BY name')
tables = c.fetchall()
print('\nUpcoming tables in database:')
for t in tables:
    # Get column info
    c.execute(f'PRAGMA table_info({t[0]})')
    columns = c.fetchall()
    print(f'\n  {t[0]}:')
    for col in columns:
        print(f'    - {col[1]} ({col[2]})')

conn.close()
print('\nDone! Tables are ready for upcoming race data.')
