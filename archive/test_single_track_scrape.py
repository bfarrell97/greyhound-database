"""Test scraping a single track to debug database save"""
from upcoming_betting_scraper import UpcomingBettingScraper
import sqlite3

scraper = UpcomingBettingScraper()

# Fetch just Nowra
print("Fetching Nowra races...")
races = scraper.fetch_upcoming_betting_races('2025-12-08')

# Filter to just Nowra
nowra_races = [r for r in races if 'Nowra' in r['track_name']]
print(f"Found {len(nowra_races)} Nowra races")

# Save to database
print("\nSaving to database...")
scraper.save_upcoming_betting_races_to_db(nowra_races)

# Check what's in database
conn = sqlite3.connect('greyhound_racing.db')
c = conn.cursor()

c.execute('SELECT COUNT(*) FROM UpcomingBettingRaces WHERE TrackName LIKE "%Nowra%"')
print(f"\nRaces in DB: {c.fetchone()[0]}")

c.execute('SELECT COUNT(*) FROM UpcomingBettingRunners ur JOIN UpcomingBettingRaces ubr ON ur.UpcomingBettingRaceID = ubr.UpcomingBettingRaceID WHERE ubr.TrackName LIKE "%Nowra%"')
print(f"Runners in DB: {c.fetchone()[0]}")

c.execute('SELECT COUNT(*) FROM UpcomingBettingRunners ur JOIN UpcomingBettingRaces ubr ON ur.UpcomingBettingRaceID = ubr.UpcomingBettingRaceID WHERE ubr.TrackName LIKE "%Nowra%" AND ur.CurrentOdds IS NOT NULL')
print(f"Runners with odds: {c.fetchone()[0]}")

c.execute('SELECT GreyhoundName, BoxNumber, CurrentOdds FROM UpcomingBettingRunners ur JOIN UpcomingBettingRaces ubr ON ur.UpcomingBettingRaceID = ubr.UpcomingBettingRaceID WHERE ubr.TrackName LIKE "%Nowra%" AND ubr.RaceNumber = 1')
print(f"\nNowra R1 runners:")
for row in c.fetchall():
    print(f"  {row[0]:<30} Box {row[1]}  Odds: {row[2]}")

conn.close()
