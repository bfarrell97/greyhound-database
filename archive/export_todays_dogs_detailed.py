"""
DETAILED MODEL FIGURES: Export all calculations for today's dogs
Shows pace, form, and weighted score components
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

DB_PATH = 'greyhound_racing.db'

# Get today's date
today = datetime.now().strftime('%Y-%m-%d')

print(f"\n{'='*120}")
print(f"DETAILED MODEL FIGURES - {today}")
print(f"{'='*120}")

conn = sqlite3.connect(DB_PATH)

# Get all upcoming dogs with detailed pace and form calculations
query = f"""
WITH dog_pace_history AS (
    SELECT 
        ge.GreyhoundID,
        g.GreyhoundName,
        rm.MeetingDate,
        (ge.FinishTimeBenchmarkLengths + rm.MeetingAvgBenchmarkLengths) as TotalFinishBench,
        ge.FinishTimeBenchmarkLengths,
        rm.MeetingAvgBenchmarkLengths,
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
        COUNT(DISTINCT CASE WHEN RaceNum <= 5 THEN MeetingDate END) as PacesUsed,
        MAX(CASE WHEN RaceNum = 1 THEN TotalFinishBench END) as MostRecentPace,
        AVG(CASE WHEN RaceNum <= 3 THEN TotalFinishBench END) as Best3RacePace,
        GROUP_CONCAT(CASE WHEN RaceNum <= 5 THEN ROUND(TotalFinishBench, 2) END, '|') as Last5Paces
    FROM dog_pace_history
    GROUP BY GreyhoundID
    HAVING PacesUsed >= 5
),

dog_recent_form AS (
    SELECT 
        ge.GreyhoundID,
        (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as IsWinner,
        ge.Position,
        ge.StartingPrice,
        rm.MeetingDate,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE ge.Position IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
      AND ge.StartingPrice IS NOT NULL
),

dog_form_last_5 AS (
    SELECT 
        GreyhoundID,
        SUM(CASE WHEN IsWinner = 1 THEN 1 ELSE 0 END) as RecentWins,
        COUNT(*) as RecentRaces,
        ROUND(100.0 * SUM(CASE WHEN IsWinner = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as RecentWinRate,
        GROUP_CONCAT(CASE WHEN RaceNum <= 5 THEN (CASE WHEN IsWinner = 1 THEN 'W' ELSE Position END) END, '') as Last5Results
    FROM dog_recent_form
    WHERE RaceNum <= 5
    GROUP BY GreyhoundID
)

SELECT 
    ubr_race.TrackName,
    ubr_race.RaceNumber,
    ubr_race.Distance,
    ubr.BoxNumber,
    ubr.GreyhoundName,
    ROUND(dpa.HistoricalPaceAvg, 4) as HistoricalPaceAvg,
    dpa.PacesUsed,
    ROUND(dpa.MostRecentPace, 4) as MostRecentPace,
    ROUND(dpa.Best3RacePace, 4) as Best3RacePace,
    dpa.Last5Paces,
    COALESCE(dfl.RecentWins, 0) as RecentWins,
    COALESCE(dfl.RecentRaces, 0) as RecentRaces,
    COALESCE(dfl.RecentWinRate, 0) as RecentWinRate,
    dfl.Last5Results,
    ubr.CurrentOdds as StartingPrice
FROM UpcomingBettingRunners ubr
JOIN UpcomingBettingRaces ubr_race ON ubr.UpcomingBettingRaceID = ubr_race.UpcomingBettingRaceID
LEFT JOIN dog_pace_avg dpa ON dpa.GreyhoundName = ubr.GreyhoundName
LEFT JOIN dog_form_last_5 dfl ON dfl.GreyhoundID = (
    SELECT GreyhoundID FROM Greyhounds WHERE GreyhoundName = ubr.GreyhoundName LIMIT 1
)
WHERE ubr_race.MeetingDate = '{today}'
  AND ubr.CurrentOdds >= 1.50
  AND ubr.CurrentOdds <= 5.00
ORDER BY ubr_race.TrackName, ubr_race.RaceNumber, ubr.BoxNumber
"""

print(f"\nLoading upcoming dogs for {today}...")
df = pd.read_sql_query(query, conn)
conn.close()

if len(df) == 0:
    print(f"No upcoming races for {today}")
    exit()

print(f"Loaded {len(df)} dogs")

# Clean data
df['HistoricalPaceAvg'] = pd.to_numeric(df['HistoricalPaceAvg'], errors='coerce')
df['RecentWinRate'] = pd.to_numeric(df['RecentWinRate'], errors='coerce').fillna(0)
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['MostRecentPace'] = pd.to_numeric(df['MostRecentPace'], errors='coerce')
df['Best3RacePace'] = pd.to_numeric(df['Best3RacePace'], errors='coerce')

# Calculate scores
pace_min = df['HistoricalPaceAvg'].min()
pace_max = df['HistoricalPaceAvg'].max()

# Handle edge case where all values are the same
if pace_max - pace_min == 0 or pd.isna(pace_max) or pd.isna(pace_min):
    df['PaceScore'] = 0.5
else:
    df['PaceScore'] = (df['HistoricalPaceAvg'] - pace_min) / (pace_max - pace_min)

df['FormScore'] = df['RecentWinRate'] / 100.0
df['WeightedScore'] = (df['PaceScore'] * 0.7) + (df['FormScore'] * 0.3)

# Select and rename columns for export
export_df = df[[
    'TrackName',
    'RaceNumber',
    'Distance',
    'BoxNumber',
    'GreyhoundName',
    'HistoricalPaceAvg',
    'PacesUsed',
    'MostRecentPace',
    'Best3RacePace',
    'Last5Paces',
    'RecentWins',
    'RecentRaces',
    'RecentWinRate',
    'Last5Results',
    'StartingPrice',
    'PaceScore',
    'FormScore',
    'WeightedScore'
]].copy()

export_df.columns = [
    'Track',
    'Race#',
    'Distance',
    'Box',
    'Dog Name',
    'HistoricalPaceAvg',
    'NumPaces',
    'MostRecentPace',
    'Best3RacePace',
    'Last5PaceValues',
    'RecentWins',
    'RecentRaces',
    'WinRate%',
    'Last5Results',
    'StartingPrice',
    'PaceScore(0-1)',
    'FormScore(0-1)',
    'WeightedScore(0-1)'
]

# Round numeric columns
numeric_cols = ['HistoricalPaceAvg', 'MostRecentPace', 'Best3RacePace', 'WinRate%', 
                'StartingPrice', 'PaceScore(0-1)', 'FormScore(0-1)', 'WeightedScore(0-1)']
for col in numeric_cols:
    if col in export_df.columns:
        export_df[col] = export_df[col].round(4)

# Sort by weighted score descending
export_df = export_df.sort_values('WeightedScore(0-1)', ascending=False)

# Export to CSV
csv_filename = f'detailed_model_figures_{today}_v2.csv'
export_df.to_csv(csv_filename, index=False)

print(f"\n[OK] Exported {len(export_df)} dogs to {csv_filename}")
print(f"\n{'='*120}")
print("PREVIEW OF DATA (Top 20):")
print(f"{'='*120}\n")

# Display in readable format
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(export_df.head(20).to_string(index=False))

print(f"\n{'='*120}")
print("SUMMARY STATISTICS")
print(f"{'='*120}")
print(f"Total Dogs: {len(export_df)}")
print(f"Weighted Score Range: {export_df['WeightedScore(0-1)'].min():.4f} to {export_df['WeightedScore(0-1)'].max():.4f}")
print(f"Average Weighted Score: {export_df['WeightedScore(0-1)'].mean():.4f}")
print(f"Average Starting Price: ${export_df['StartingPrice'].mean():.2f}")
print(f"Dogs with Score >= 0.6: {len(export_df[export_df['WeightedScore(0-1)'] >= 0.6])}")
print(f"\nFile saved to: {csv_filename}")
print(f"{'='*120}\n")
