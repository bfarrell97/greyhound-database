"""
Show only HistoricalPace column for tomorrow's races
Uses EXACT same methodology as +13% ROI validation
"""
import sqlite3
import pandas as pd
from datetime import datetime

conn = sqlite3.connect('greyhound_racing.db')

# Get latest upcoming race date
latest_date = pd.read_sql_query(
    "SELECT DISTINCT MeetingDate FROM UpcomingBettingRaces ORDER BY MeetingDate DESC LIMIT 1",
    conn
)['MeetingDate'].values[0]

# Get all upcoming runners with historical pace calculated using exact +13% ROI methodology
query = """
WITH dog_pace_history AS (
    SELECT 
        ge.GreyhoundID,
        g.GreyhoundName,
        rm.MeetingDate,
        (ge.FinishTimeBenchmarkLengths + rm.MeetingAvgBenchmarkLengths) as TotalFinishBench,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE ge.FinishTimeBenchmarkLengths IS NOT NULL
      AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
      AND ge.Position IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
),

dog_pace_avg AS (
    SELECT 
        GreyhoundID,
        GreyhoundName,
        AVG(CASE WHEN RaceNum <= 5 THEN TotalFinishBench END) as HistoricalPaceAvg,
        COUNT(DISTINCT CASE WHEN RaceNum <= 5 THEN MeetingDate END) as RacesUsed
    FROM dog_pace_history
    GROUP BY GreyhoundID
    HAVING RacesUsed >= 5
)

SELECT 
    ubr_race.TrackName,
    ubr_race.RaceNumber,
    ubr_race.Distance,
    ubr.BoxNumber,
    ubr.GreyhoundName,
    dpa.HistoricalPaceAvg as HistoricalPace,
    ubr.CurrentOdds
FROM UpcomingBettingRunners ubr
JOIN UpcomingBettingRaces ubr_race ON ubr.UpcomingBettingRaceID = ubr_race.UpcomingBettingRaceID
LEFT JOIN dog_pace_avg dpa ON ubr.GreyhoundName = dpa.GreyhoundName
WHERE ubr_race.MeetingDate = ?
ORDER BY ubr_race.TrackName, ubr_race.RaceNumber, ubr.BoxNumber
"""

runners_df = pd.read_sql_query(query, conn, params=(latest_date,))

conn.close()

# Save to CSV
csv_file = f"historical_pace_{latest_date}.csv"
runners_df.to_csv(csv_file, index=False, columns=['TrackName', 'RaceNumber', 'Distance', 'BoxNumber', 'GreyhoundName', 'HistoricalPace', 'CurrentOdds'])
print(f"Saved to: {csv_file}\n")

# Display
print(f"\n{'='*120}")
print(f"HISTORICAL PACE - {latest_date} (Using +13% ROI Methodology)")
print(f"{'='*120}\n")

current_race = None
for _, row in runners_df.iterrows():
    race_key = f"{row['TrackName']} R{int(row['RaceNumber'])} ({int(row['Distance'])}m)"
    if race_key != current_race:
        if current_race:
            print()
        print(race_key)
        print("-" * 120)
        current_race = race_key
    
    box = int(row['BoxNumber']) if row['BoxNumber'] else '?'
    name = row['GreyhoundName']
    pace = row['HistoricalPace']
    odds = f"${row['CurrentOdds']:.2f}" if row['CurrentOdds'] and row['CurrentOdds'] > 0 else 'N/A'
    
    if pd.isna(pace):
        pace_str = "+nan"
    else:
        pace_str = f"{pace:+.2f}"
    
    print(f"  Box {box:2} | {name:25} | Pace: {pace_str:>7} | Odds: {odds:>7}")

print(f"\n{'='*120}\n")
