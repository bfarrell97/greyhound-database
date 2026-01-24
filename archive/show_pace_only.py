"""
Show historical pace data for all upcoming runners
Simplified view - just pace data, no odds or other information
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect('greyhound_racing.db')

# Get latest upcoming race date
query = "SELECT DISTINCT MeetingDate FROM UpcomingBettingRaces ORDER BY MeetingDate DESC LIMIT 1"
latest_date = pd.read_sql_query(query, conn)['MeetingDate'].values[0]

print(f"\n{'='*100}")
print(f"HISTORICAL PACE DATA - {latest_date}")
print(f"{'='*100}\n")

# Get all races
races_query = """
SELECT DISTINCT 
    ubr.TrackName,
    ubr.MeetingDate,
    ubr.RaceNumber,
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
    
    # Get runners
    runners_query = """
    SELECT 
        ubr.BoxNumber,
        ubr.GreyhoundName
    FROM UpcomingBettingRunners ubr
    JOIN UpcomingBettingRaces ubr_race ON ubr.UpcomingBettingRaceID = ubr_race.UpcomingBettingRaceID
    WHERE ubr_race.TrackName = ? AND ubr_race.MeetingDate = ? AND ubr_race.RaceNumber = ?
    ORDER BY ubr.BoxNumber
    """
    
    runners = pd.read_sql_query(runners_query, conn, params=(track, date, race_num))
    
    if runners.empty:
        continue
    
    print(f"{track:18} R{race_num:2d} ({distance}m) - {len(runners)} runners")
    print("-" * 100)
    
    for _, runner in runners.iterrows():
        box = int(runner['BoxNumber']) if runner['BoxNumber'] else '?'
        name = runner['GreyhoundName']
        
        # Get last 5 races - use FinishTimeBenchmarkLengths from GreyhoundEntries
        pace_query = """
        SELECT 
            ge.FinishTimeBenchmarkLengths,
            rm.MeetingDate,
            r.TrackName as RaceTrack,
            r.RaceNumber,
            r.Distance,
            ge.Position,
            rm.MeetingAvgBenchmarkLengths
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE g.Name = ? 
            AND rm.MeetingDate < ?
            AND ge.Position IS NOT NULL
            AND ge.Position <= 6
            AND ge.FinishTimeBenchmarkLengths IS NOT NULL
        ORDER BY rm.MeetingDate DESC
        LIMIT 5
        """
        
        pace_data = pd.read_sql_query(pace_query, conn, params=(name, date))
        
        if pace_data.empty:
            print(f"  Box {box:2} | {name:25} | NO RECENT RACES")
        else:
            # Calculate pace: average of FinishTimeBenchmarkLengths
            pace_values = pace_data['FinishTimeBenchmarkLengths']
            avg_pace = pace_values.mean()
            
            # Show recent races
            recent = ", ".join([
                f"{row['RaceTrack'][:3].upper()} {row['MeetingDate'][-5:]} ({row['FinishTimeBenchmarkLengths']:+.2f}L #{int(row['Position'])})"
                for _, row in pace_data.iterrows()
            ])
            
            print(f"  Box {box:2} | {name:25} | Pace: {avg_pace:+7.2f}L | {recent}")
    
    print()

conn.close()

print(f"{'='*100}")
print("PACE VALUES: Positive = faster than benchmark | Negative = slower than benchmark")
print(f"{'='*100}\n")
