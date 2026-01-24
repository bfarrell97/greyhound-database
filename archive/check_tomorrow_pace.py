"""
Check historical pace data for tomorrow's runners
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

# Connect to database
conn = sqlite3.connect('greyhound_racing.db')
conn.row_factory = sqlite3.Row

tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

# First, check what tables exist
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print(f"Available tables: {', '.join(tables)}\n")

# Check what dates we have
cursor.execute("SELECT DISTINCT MeetingDate FROM UpcomingBettingRaces ORDER BY MeetingDate DESC LIMIT 1")
latest_date = cursor.fetchone()
if latest_date:
    check_date = latest_date[0]
    print(f"Tomorrow's date requested: {tomorrow}")
    print(f"Latest data available: {check_date}")
    if check_date != tomorrow:
        print(f"Using available date: {check_date}\n")
        tomorrow = check_date
else:
    print("No upcoming race data found\n")

# If UpcomingBettingRaces exists, use it; otherwise look for upcoming races in Races table
has_upcoming = 'UpcomingBettingRaces' in tables

if has_upcoming:
    print("Found UpcomingBettingRaces table")
    query = """
    SELECT DISTINCT 
        ubr.TrackName,
        ubr.MeetingDate,
        ubr.RaceNumber,
        ubr.Distance
    FROM UpcomingBettingRaces ubr
    WHERE ubr.MeetingDate = ?
    ORDER BY ubr.TrackName, ubr.RaceNumber
    """
    races_df = pd.read_sql_query(query, conn, params=(tomorrow,))
    table_name = "UpcomingBettingRaces"
else:
    print("UpcomingBettingRaces not found, looking in Races table")
    query = """
    SELECT DISTINCT 
        r.TrackName,
        DATE(r.RaceDate) as MeetingDate,
        r.RaceNumber,
        r.Distance
    FROM Races r
    WHERE DATE(r.RaceDate) = ?
    ORDER BY r.TrackName, r.RaceNumber
    """
    races_df = pd.read_sql_query(query, conn, params=(tomorrow,))
    table_name = "Races"

if races_df.empty:
    print(f"No races found for {tomorrow}")
    conn.close()
    exit()

print(f"\n{'='*120}")
print(f"HISTORICAL PACE DATA - {tomorrow}")
print(f"{'='*120}\n")

# For each race, get the runners and their pace data
for _, race in races_df.iterrows():
    track = race['TrackName']
    date = race['MeetingDate'] if 'MeetingDate' in race.keys() else race['RaceDate']
    race_num = int(race['RaceNumber'])
    distance = int(race['Distance'])
    
    # Get runners for this race
    if has_upcoming:
        runners_query = """
        SELECT DISTINCT
            ubr.GreyhoundID,
            ubr.GreyhoundName,
            ubr.TrailingOdds
        FROM UpcomingBettingRunners ubr
        WHERE ubr.TrackName = ? AND ubr.MeetingDate = ? AND ubr.RaceNumber = ?
        ORDER BY ubr.GreyhoundName
        """
    else:
        runners_query = """
        SELECT DISTINCT
            ge.GreyhoundID,
            g.Name as GreyhoundName,
            NULL as TrailingOdds
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        WHERE r.TrackName = ? AND DATE(r.RaceDate) = ? AND r.RaceNumber = ?
        ORDER BY g.Name
        """
    
    runners = pd.read_sql_query(runners_query, conn, params=(track, date, race_num))
    
    print(f"{track:15} | {date} | Race {race_num:2d} | {distance}m | {len(runners)} runners")
    print("-" * 120)
    
    if runners.empty:
        print("  (No runners in database yet)")
        continue
    
    # For each runner, calculate historical pace
    for _, runner in runners.iterrows():
        dog_id = runner['GreyhoundID']
        dog_name = runner['GreyhoundName']
        odds = runner['TrailingOdds'] if runner['TrailingOdds'] else 'N/A'
        
        # Get last 5 completed races for pace calculation
        pace_query = """
        SELECT 
            r.FinishTimeBenchmarkLengths,
            rm.MeetingAvgBenchmarkLengths,
            r.RaceDate,
            r.TrackName as RaceTrack,
            r.RaceNumber
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.RaceID = rm.RaceID
        WHERE ge.GreyhoundID = ? 
            AND DATE(r.RaceDate) < ?
            AND ge.FinishPosition IS NOT NULL
            AND ge.FinishPosition <= 6
        ORDER BY r.RaceDate DESC
        LIMIT 5
        """
        
        pace_data = pd.read_sql_query(pace_query, conn, params=(dog_id, date))
        
        if pace_data.empty:
            pace = "N/A"
            recent_races = "No recent races"
        else:
            # Calculate pace: average of (FinishTime + MeetingAvg) over last 5 races
            pace_values = pace_data['FinishTimeBenchmarkLengths'] + pace_data['MeetingAvgBenchmarkLengths']
            pace = pace_values.mean()
            
            recent_races = f"{len(pace_data)} races: " + ", ".join([
                f"{row['RaceTrack']} {row['RaceDate'][-5:]}"
                for _, row in pace_data.iterrows()
            ])
        
        if isinstance(pace, str):
            print(f"  {dog_name:25} | Pace: {pace:>7}  | Odds: ${odds:>5} | {recent_races}")
        else:
            print(f"  {dog_name:25} | Pace: {pace:7.2f}  | Odds: ${odds:>5} | {recent_races}")
    
    print()

conn.close()

print(f"{'='*120}")
print(f"PACE INTERPRETATION:")
print(f"  Positive pace = faster than benchmark (better)")
print(f"  Negative pace = slower than benchmark (worse)")
print(f"  N/A = insufficient race history in database")
print(f"{'='*120}\n")
