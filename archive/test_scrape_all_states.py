"""Test scraping all states"""
from upcoming_betting_scraper import UpcomingBettingScraper

scraper = UpcomingBettingScraper()

# Fetch races for today
races = scraper.fetch_upcoming_betting_races('2025-12-08')

print(f"\n{'='*80}")
print(f"SCRAPING RESULTS")
print(f"{'='*80}")
print(f"\nTotal races fetched: {len(races)}")

# Group by track
tracks = {}
for race in races:
    track = race['track_name']
    if track not in tracks:
        tracks[track] = 0
    tracks[track] += 1

print(f"\nRaces by track:")
print("-" * 60)
for track, count in sorted(tracks.items()):
    print(f"  {track:<30} {count:>3} races")

# Save to database
print(f"\n{'='*80}")
print("SAVING TO DATABASE")
print(f"{'='*80}")
scraper.save_upcoming_betting_races_to_db(races)
