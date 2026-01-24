"""
Fast Track-scaled pace & form model backtest (Sept-Nov 2025)
Process all data in bulk for speed
Export detailed CSV with all bets for analysis
"""

import sqlite3
import pandas as pd
from datetime import datetime
import os

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

# Track tier definitions
METRO_TRACKS = {
    'Wentworth Park', 'Albion Park', 'Angle Park',
    'Sandown Park', 'The Meadows', 'Cannington'
}

PROVINCIAL_TRACKS = {
    'Richmond', 'Richmond Straight', 'Nowra', 'The Gardens', 'Bulli',
    'Dapto', 'Maitland', 'Goulburn', 'Ipswich', 'Q Straight',
    'Q1 Lakeside', 'Q2 Parklands', 'Gawler',
    'Ballarat', 'Bendigo', 'Geelong', 'Sale', 'Cranbourne', 'Warrnambool', 'Mandurah'
}

TRACK_SCALING = {'metro': 4.0, 'provincial': 2.0, 'country': 1.0}

def get_track_tier(track_name):
    if track_name in METRO_TRACKS:
        return 'metro'
    elif track_name in PROVINCIAL_TRACKS:
        return 'provincial'
    else:
        return 'country'

progress("=" * 100)
progress("TRACK-SCALED BACKTEST (FAST)")
progress("=" * 100)
progress("Processing Sept-Nov 2025 in bulk...\n")

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# ============================================================================
# IMPORTANT: We need to calculate historical metrics BEFORE the test period
# to avoid look-ahead bias. Use data < 2025-09-01 for pace/form calculations.
# ============================================================================
progress("Fetching historical pace/form data (before Sept 2025)...", indent=1)

# Step 1: Get historical pace metrics (data BEFORE Sept 1, 2025)
pace_query = """
WITH dog_pace_history_raw AS (
    SELECT 
        ge.GreyhoundID,
        g.GreyhoundName,
        rm.MeetingDate,
        t.TrackName,
        ge.FinishTimeBenchmarkLengths,
        rm.MeetingAvgBenchmarkLengths,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.FinishTimeBenchmarkLengths IS NOT NULL
      AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
      AND ge.Position IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
      AND rm.MeetingDate < '2025-09-01'
      AND t.TrackName NOT LIKE '%NZ%'
      AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
)
SELECT 
    GreyhoundID,
    GreyhoundName,
    AVG(CASE WHEN RaceNum <= 5 THEN (FinishTimeBenchmarkLengths + MeetingAvgBenchmarkLengths) END) as RawPaceAvg,
    COUNT(DISTINCT CASE WHEN RaceNum <= 5 THEN MeetingDate END) as PacesUsed
FROM dog_pace_history_raw
GROUP BY GreyhoundID, GreyhoundName
HAVING PacesUsed >= 5
"""

pace_df = pd.read_sql_query(pace_query, conn)
progress(f"  Found {len(pace_df):,} dogs with valid pace history", indent=1)

# Step 2: Get historical form metrics (data BEFORE Sept 1, 2025)
form_query = """
WITH dog_recent_form_raw AS (
    SELECT 
        ge.GreyhoundID,
        (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as IsWinner,
        rm.MeetingDate,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.Position IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
      AND rm.MeetingDate < '2025-09-01'
      AND t.TrackName NOT LIKE '%NZ%'
      AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
)
SELECT 
    GreyhoundID,
    SUM(CASE WHEN RaceNum <= 5 AND IsWinner = 1 THEN 1 ELSE 0 END) as RawWins,
    COUNT(CASE WHEN RaceNum <= 5 THEN 1 END) as FormRaces
FROM dog_recent_form_raw
GROUP BY GreyhoundID
"""

form_df = pd.read_sql_query(form_query, conn)
progress(f"  Found {len(form_df):,} dogs with form history", indent=1)

# Step 3: Get all races in test period (Sept-Nov 2025)
progress("Fetching races in test period (Sept-Nov 2025)...", indent=1)

races_query = """
SELECT 
    ge.GreyhoundID,
    g.GreyhoundName,
    ge.StartingPrice,
    t.TrackName,
    DATE(rm.MeetingDate) as RaceDate,
    (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as ActualWinner
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
  AND ge.StartingPrice IS NOT NULL
  AND rm.MeetingDate >= '2025-09-01'
  AND rm.MeetingDate < '2025-12-01'
  AND t.TrackName NOT LIKE '%NZ%'
  AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
"""

df = pd.read_sql_query(races_query, conn)
conn.close()

progress(f"  Found {len(df):,} races in test period", indent=1)

# Merge historical pace and form with races
df = df.merge(pace_df[['GreyhoundID', 'RawPaceAvg']], on='GreyhoundID', how='left')
df = df.merge(form_df[['GreyhoundID', 'RawWins', 'FormRaces']], on='GreyhoundID', how='left')

# Only keep dogs with valid historical data
df = df.dropna(subset=['RawPaceAvg'])
progress(f"  {len(df):,} races have valid historical pace data\n", indent=1)

# Convert to numeric
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df = df.dropna(subset=['StartingPrice'])

# Add track tier
df['TrackTier'] = df['TrackName'].apply(get_track_tier)

# Apply track scaling
df['ScaledPace'] = df.apply(
    lambda row: row['RawPaceAvg'] * TRACK_SCALING[row['TrackTier']] if pd.notna(row['RawPaceAvg']) else 0,
    axis=1
)

df['ScaledWins'] = df.apply(
    lambda row: row['RawWins'] * TRACK_SCALING[row['TrackTier']] if pd.notna(row['RawWins']) else 0,
    axis=1
)

# Normalize scores
pace_min = df[df['ScaledPace'] > 0]['ScaledPace'].min()
pace_max = df['ScaledPace'].max()

if pace_max - pace_min > 0:
    df['PaceScore'] = (df['ScaledPace'] - pace_min) / (pace_max - pace_min)
else:
    df['PaceScore'] = 0.5

df['FormRateScaled'] = df.apply(
    lambda row: (row['ScaledWins'] / row['FormRaces'] * 100) if row['FormRaces'] > 0 else 0,
    axis=1
)
df['FormScore'] = df['FormRateScaled'] / 100.0

# Weighted score
df['WeightedScore'] = (df['PaceScore'] * 0.7) + (df['FormScore'] * 0.3)

# Calculate P&L
df['Profit'] = df.apply(lambda row: row['StartingPrice'] - 1 if row['ActualWinner'] == 1 else -1, axis=1)

# Filter to high confidence AND odds $1.50-$5.00
high_conf = df[(df['WeightedScore'] >= 0.80) & (df['StartingPrice'] >= 1.50) & (df['StartingPrice'] <= 5.00)].copy().sort_values('RaceDate')

progress(f"Calculating stats...\n", indent=1)

# Overall stats
total_bets = len(high_conf)
total_wins = high_conf['ActualWinner'].sum()
strike_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
avg_odds = high_conf['StartingPrice'].mean()
roi = ((total_wins * avg_odds) - total_bets) / total_bets * 100 if total_bets > 0 else 0

progress("=" * 100)
progress("TRACK-SCALED MODEL RESULTS (70/30 @ 0.80)")
progress("=" * 100)
progress(f"\nTotal Bets: {total_bets}", indent=1)
progress(f"Total Wins: {total_wins}", indent=1)
progress(f"Strike Rate: {strike_rate:.1f}%", indent=1)
progress(f"Average Odds: ${avg_odds:.2f}", indent=1)
progress(f"ROI: {roi:+.1f}%", indent=1)

# Price brackets
progress(f"\n" + "=" * 100)
progress("BREAKDOWN BY PRICE BRACKETS")
progress("=" * 100)

price_brackets = [
    (1.50, 2.00, "$1.50-$2.00"),
    (2.00, 2.50, "$2.00-$2.50"),
    (2.50, 3.00, "$2.50-$3.00"),
    (3.00, 4.00, "$3.00-$4.00"),
    (4.00, 5.00, "$4.00-$5.00"),
    (5.00, 6.00, "$5.00-$6.00"),
    (6.00, 8.00, "$6.00-$8.00"),
    (8.00, 10.00, "$8.00-$10.00"),
    (10.00, 15.00, "$10.00-$15.00"),
    (15.00, 20.00, "$15.00-$20.00"),
]

for min_price, max_price, label in price_brackets:
    bracket_data = high_conf[(high_conf['StartingPrice'] >= min_price) & (high_conf['StartingPrice'] < max_price)]
    if len(bracket_data) == 0:
        continue
    
    wins = bracket_data['ActualWinner'].sum()
    strike = (wins / len(bracket_data)) * 100
    avg_odds_bracket = bracket_data['StartingPrice'].mean()
    roi_bracket = ((wins * avg_odds_bracket) - len(bracket_data)) / len(bracket_data) * 100
    
    progress(f"\n{label}:", indent=1)
    progress(f"  Bets: {len(bracket_data)} | Wins: {wins} | Strike: {strike:.1f}% | Avg Odds: ${avg_odds_bracket:.2f} | ROI: {roi_bracket:+.1f}%", indent=2)

# Export detailed CSV
progress(f"\n" + "=" * 100)
progress("EXPORTING DETAILED BET LIST...")
progress("=" * 100)

export_cols = [
    'RaceDate', 'GreyhoundName', 'TrackName', 'TrackTier',
    'StartingPrice', 'WeightedScore', 'PaceScore', 'FormScore',
    'RawPaceAvg', 'FormRateScaled', 'ActualWinner', 'Profit'
]

export_df = high_conf[export_cols].copy()
export_df['RaceDate'] = pd.to_datetime(export_df['RaceDate'])
export_df = export_df.sort_values('RaceDate')

csv_file = 'backtest_track_scaled_all_bets.csv'
export_df.to_csv(csv_file, index=False)
progress(f"\nExported {len(export_df)} bets to {csv_file}", indent=1)

# Show sample
progress(f"\nFirst 20 bets:", indent=1)
for idx, row in export_df.head(20).iterrows():
    result = "✓ WIN" if row['ActualWinner'] == 1 else "✗ LOSS"
    progress(f"  {row['RaceDate'].strftime('%Y-%m-%d')} | {row['GreyhoundName']:20} | ${row['StartingPrice']:5.2f} | Score: {row['WeightedScore']:.3f} | {result}", indent=2)

progress(f"\n" + "=" * 100)
progress("DONE!")
progress("=" * 100)
