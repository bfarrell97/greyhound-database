"""
Check historical pace data for upcoming runners
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

# Connect to database
conn = sqlite3.connect('greyhound_racing.db')

# Get the latest upcoming race date
query = "SELECT DISTINCT MeetingDate FROM UpcomingBettingRaces ORDER BY MeetingDate DESC LIMIT 1"
latest_date = pd.read_sql_query(query, conn)['MeetingDate'].values[0]

print(f"\n{'='*120}")
print(f"HISTORICAL PACE DATA - {latest_date}")
print(f"{'='*120}\n")

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

for _, race in races_df.iterrows():
    track = race['TrackName']
    date = race['MeetingDate']
    race_num = int(race['RaceNumber'])
    distance = int(race['Distance'])
    race_time = race['RaceTime'] or 'TBA'
    
    # Get runners for this race
    runners_query = """
    SELECT DISTINCT
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
    
    print(f"{track:18} | {date} | Race {race_num:2d} | {distance}m @ {race_time:5} | {len(runners)} runners")
    print("-" * 120)
    
    # For each runner, get historical pace
    for _, runner in runners.iterrows():
        box = int(runner['BoxNumber']) if runner['BoxNumber'] else '?'
        name = runner['GreyhoundName']
        form = runner['Form'] if runner['Form'] else '-'
        odds = runner['CurrentOdds'] if runner['CurrentOdds'] else 'N/A'
        weight = f"{int(runner['Weight'])}" if runner['Weight'] else 'N/A'
        
        # Get historical pace for this dog - need to find by name in completed races
        pace_query = """
        SELECT 
            r.FinishTimeBenchmarkLengths,
            rm.MeetingAvgBenchmarkLengths,
            r.RaceDate,
            r.TrackName as RaceTrack,
            r.RaceNumber,
            ge.FinishPosition
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.RaceID = rm.RaceID
        WHERE g.Name = ? 
            AND DATE(r.RaceDate) < ?
            AND ge.FinishPosition IS NOT NULL
            AND ge.FinishPosition <= 6
        ORDER BY r.RaceDate DESC
        LIMIT 5
        """
        
        pace_data = pd.read_sql_query(pace_query, conn, params=(name, date))
        
        if pace_data.empty:
            pace = "N/A"
            recent_races = "No recent races"
        else:
            # Calculate pace: average of (FinishTime + MeetingAvg) over last 5 races
            pace_values = pace_data['FinishTimeBenchmarkLengths'] + pace_data['MeetingAvgBenchmarkLengths']
            pace = pace_values.mean()
            
            recent_races = f"{len(pace_data)} races: " + ", ".join([
                f"{row['RaceTrack'][:3].upper()} {row['RaceDate'][-5:]} (#{int(row['FinishPosition'])})"
                for _, row in pace_data.iterrows()
            ])
        
        if isinstance(pace, str):
            print(f"  Box {box:2} | {name:25} | Form: {form:4} | Pace: {pace:>7}  | Odds: ${odds:>6} | {recent_races}")
        else:
            confidence = "[HIGH]" if pace > 0.5 else "[MID]" if pace > -0.5 else "[LOW]"
            print(f"  Box {box:2} | {name:25} | Form: {form:4} | Pace: {pace:7.2f}  | Odds: ${odds:>6} | {confidence}")
    
    print()

conn.close()

print(f"{'='*120}")
print(f"PACE INTERPRETATION:")
print(f"  [HIGH] pace > +0.5  = significantly faster than benchmark (strong)")
print(f"  [MID]  pace -0.5 to +0.5 = near benchmark (neutral)")
print(f"  [LOW]  pace < -0.5  = slower than benchmark (weak)")
print(f"  N/A = insufficient race history in database")
print(f"{'='*120}\n")
