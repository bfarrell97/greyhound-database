"""
Show tomorrow's upcoming runners with basic info
"""
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('greyhound_racing.db')

# Get the latest upcoming race date
query = "SELECT DISTINCT MeetingDate FROM UpcomingBettingRaces ORDER BY MeetingDate DESC LIMIT 1"
latest_date = pd.read_sql_query(query, conn)['MeetingDate'].values[0]

print(f"\n{'='*130}")
print(f"TOMORROW'S RUNNERS - {latest_date}")
print(f"{'='*130}\n")

# Get all races for that date
races_query = """
SELECT DISTINCT 
    ubr.TrackName,
    ubr.MeetingDate,
    ubr.RaceNumber,
    ubr.RaceTime,
    ubr.Distance
FROM UpcomingBettingRaces ubr
WHERE ubr.MeetingDate = ?
ORDER BY ubr.TrackName, ubr.RaceNumber
"""

races_df = pd.read_sql_query(races_query, conn, params=(latest_date,))

runner_count = 0

for _, race in races_df.iterrows():
    track = race['TrackName']
    date = race['MeetingDate']
    race_num = int(race['RaceNumber'])
    distance = int(race['Distance'])
    race_time = race['RaceTime'] or 'TBA'
    
    # Get runners for this race
    runners_query = """
    SELECT 
        ubr.BoxNumber,
        ubr.GreyhoundName,
        ubr.Form,
        ubr.CurrentOdds,
        ubr.Weight
    FROM UpcomingBettingRunners ubr
    JOIN UpcomingBettingRaces ubr_race ON ubr.UpcomingBettingRaceID = ubr_race.UpcomingBettingRaceID
    WHERE ubr_race.TrackName = ? AND ubr_race.MeetingDate = ? AND ubr_race.RaceNumber = ?
    ORDER BY ubr.BoxNumber
    """
    
    runners = pd.read_sql_query(runners_query, conn, params=(track, date, race_num))
    
    if runners.empty:
        continue
    
    runner_count += len(runners)
    print(f"{track:18} | Race {race_num:2d} | {distance:3d}m @ {race_time:5} | {len(runners)} runners")
    print("-" * 130)
    
    # For each runner, show basic info
    for _, runner in runners.iterrows():
        box = int(runner['BoxNumber']) if runner['BoxNumber'] else '?'
        name = runner['GreyhoundName']
        form = runner['Form'] if runner['Form'] else '-'
        odds = runner['CurrentOdds'] if runner['CurrentOdds'] and runner['CurrentOdds'] > 0 else 'N/A'
        weight = f"{int(runner['Weight'])}kg" if runner['Weight'] else 'N/A'
        
        if isinstance(odds, str):
            odds_str = odds
        else:
            odds_str = f"${odds:.2f}"
        
        print(f"  Box {box:2} | {name:25} | Form: {form:6} | Odds: {odds_str:>7} | Weight: {weight:>5}")
    
    print()

conn.close()

print(f"{'='*130}")
print(f"SUMMARY: {runner_count} runners across {len(races_df)} races")
print(f"\nTo run the betting system for {latest_date}:")
print(f"  python betting_system_production.py")
print(f"{'='*130}\n")
