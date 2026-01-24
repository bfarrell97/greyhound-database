"""
Model Analysis & Optimization
Goal: 3-10 bets/day with ~30% ROI
"""

import sqlite3
import pandas as pd
from datetime import datetime

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

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
progress("MODEL ANALYSIS - Understanding bet volume")
progress("=" * 100)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Get one week of data to analyze
week_start = '2025-11-01'
week_end = '2025-11-07'

progress(f"\nAnalyzing week: {week_start} to {week_end}\n")

# Step 1: How many total races in this week?
total_races = pd.read_sql_query(f"""
SELECT COUNT(*) as cnt FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate >= '{week_start}' AND rm.MeetingDate <= '{week_end}'
""", conn)['cnt'].iloc[0]
progress(f"Total race entries in week: {total_races:,}")

# Step 2: How many with valid starting price?
with_price = pd.read_sql_query(f"""
SELECT COUNT(*) as cnt FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate >= '{week_start}' AND rm.MeetingDate <= '{week_end}'
  AND ge.StartingPrice IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
""", conn)['cnt'].iloc[0]
progress(f"With valid price & position: {with_price:,}")

# Step 3: In $2.50-$4.00 range?
in_range = pd.read_sql_query(f"""
SELECT COUNT(*) as cnt FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate >= '{week_start}' AND rm.MeetingDate <= '{week_end}'
  AND ge.StartingPrice IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
  AND CAST(ge.StartingPrice AS REAL) >= 2.5 
  AND CAST(ge.StartingPrice AS REAL) <= 4.0
""", conn)['cnt'].iloc[0]
progress(f"In $2.50-$4.00 range: {in_range:,}")

# Step 4: How many dogs have valid historical pace (5+ races)?
pace_query = f"""
WITH dog_pace_history_raw AS (
    SELECT 
        ge.GreyhoundID,
        rm.MeetingDate,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE ge.FinishTimeBenchmarkLengths IS NOT NULL
      AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
      AND rm.MeetingDate < '{week_start}'
)
SELECT COUNT(DISTINCT GreyhoundID) as cnt
FROM dog_pace_history_raw
WHERE RaceNum <= 5
GROUP BY GreyhoundID
HAVING COUNT(*) >= 5
"""
dogs_with_pace = len(pd.read_sql_query(pace_query, conn))
progress(f"Dogs with 5+ historical pace races: {dogs_with_pace:,}")

# Let's get actual candidates and see score distribution
progress("\n" + "=" * 100)
progress("Getting actual candidates for this week...")

# Get pace data
pace_df = pd.read_sql_query(f"""
WITH dog_pace_history_raw AS (
    SELECT 
        ge.GreyhoundID,
        rm.MeetingDate,
        t.TrackName,
        (ge.FinishTimeBenchmarkLengths + rm.MeetingAvgBenchmarkLengths) as TotalBench,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.FinishTimeBenchmarkLengths IS NOT NULL
      AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
      AND rm.MeetingDate < '{week_start}'
      AND t.TrackName NOT LIKE '%NZ%'
      AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
)
SELECT 
    GreyhoundID,
    AVG(CASE WHEN RaceNum <= 5 THEN TotalBench END) as RawPaceAvg,
    COUNT(DISTINCT CASE WHEN RaceNum <= 5 THEN MeetingDate END) as PacesUsed
FROM dog_pace_history_raw
GROUP BY GreyhoundID
HAVING PacesUsed >= 5
""", conn)

# Get form data
form_df = pd.read_sql_query(f"""
WITH dog_form_raw AS (
    SELECT 
        ge.GreyhoundID,
        (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as IsWinner,
        rm.MeetingDate,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE ge.Position NOT IN ('DNF', 'SCR')
      AND rm.MeetingDate < '{week_start}'
)
SELECT 
    GreyhoundID,
    SUM(CASE WHEN RaceNum <= 5 AND IsWinner = 1 THEN 1 ELSE 0 END) as RawWins,
    COUNT(CASE WHEN RaceNum <= 5 THEN 1 END) as FormRaces
FROM dog_form_raw
GROUP BY GreyhoundID
""", conn)

# Get week's races - BROADER price range first
races_df = pd.read_sql_query(f"""
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
WHERE ge.Position NOT IN ('DNF', 'SCR')
  AND ge.StartingPrice IS NOT NULL
  AND rm.MeetingDate >= '{week_start}'
  AND rm.MeetingDate <= '{week_end}'
  AND t.TrackName NOT LIKE '%NZ%'
  AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
""", conn)

conn.close()

progress(f"Week's races loaded: {len(races_df):,}")

# Merge
df = races_df.merge(pace_df, on='GreyhoundID', how='left')
df = df.merge(form_df, on='GreyhoundID', how='left')

# Convert price
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df = df.dropna(subset=['StartingPrice'])

progress(f"After price conversion: {len(df):,}")

# Check how many have pace data
with_pace = df.dropna(subset=['RawPaceAvg'])
progress(f"With valid pace data: {len(with_pace):,}")

# Now let's calculate scores for ALL dogs with pace data
df = with_pace.copy()
df['TrackTier'] = df['TrackName'].apply(get_track_tier)

# NO scaling - just raw metrics first
df['PaceRaw'] = df['RawPaceAvg']
df['FormRate'] = df.apply(
    lambda row: (row['RawWins'] / row['FormRaces'] * 100) if pd.notna(row['FormRaces']) and row['FormRaces'] > 0 else 0,
    axis=1
)

# Normalize
pace_min = df['PaceRaw'].min()
pace_max = df['PaceRaw'].max()
df['PaceScore'] = (df['PaceRaw'] - pace_min) / (pace_max - pace_min) if pace_max > pace_min else 0.5
df['FormScore'] = df['FormRate'] / 100.0

# 70/30 weighted
df['WeightedScore'] = (df['PaceScore'] * 0.7) + (df['FormScore'] * 0.3)

progress("\n" + "=" * 100)
progress("SCORE DISTRIBUTION (all dogs with pace data)")
progress("=" * 100)

# Score distribution
for threshold in [0.9, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]:
    count = len(df[df['WeightedScore'] >= threshold])
    progress(f"Score >= {threshold:.2f}: {count:,} dogs")

progress("\n" + "=" * 100)
progress("BY PRICE RANGE (Score >= 0.70)")
progress("=" * 100)

filtered = df[df['WeightedScore'] >= 0.70]

price_ranges = [
    (1.50, 2.00, "$1.50-$2.00"),
    (2.00, 2.50, "$2.00-$2.50"),
    (2.50, 3.00, "$2.50-$3.00"),
    (3.00, 4.00, "$3.00-$4.00"),
    (4.00, 5.00, "$4.00-$5.00"),
]

for min_p, max_p, label in price_ranges:
    bracket = filtered[(filtered['StartingPrice'] >= min_p) & (filtered['StartingPrice'] < max_p)]
    if len(bracket) > 0:
        wins = bracket['ActualWinner'].sum()
        strike = wins / len(bracket) * 100
        avg_odds = bracket['StartingPrice'].mean()
        roi = ((wins * avg_odds) - len(bracket)) / len(bracket) * 100
        progress(f"{label}: {len(bracket)} bets | {wins} wins | {strike:.1f}% strike | ROI: {roi:+.1f}%")
    else:
        progress(f"{label}: 0 bets")

progress("\n" + "=" * 100)
progress("TESTING DIFFERENT THRESHOLDS ($2.50-$4.00 only)")
progress("=" * 100)

price_filtered = df[(df['StartingPrice'] >= 2.50) & (df['StartingPrice'] <= 4.00)]
progress(f"Total in $2.50-$4.00 range with pace data: {len(price_filtered)}")

for threshold in [0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]:
    t_df = price_filtered[price_filtered['WeightedScore'] >= threshold]
    if len(t_df) > 0:
        wins = t_df['ActualWinner'].sum()
        strike = wins / len(t_df) * 100
        avg_odds = t_df['StartingPrice'].mean()
        roi = ((wins * avg_odds) - len(t_df)) / len(t_df) * 100
        daily = len(t_df) / 7  # 7 days in week
        progress(f"Threshold {threshold:.2f}: {len(t_df):3} bets ({daily:.1f}/day) | {wins} wins | {strike:.1f}% | ROI: {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("TESTING WIDER PRICE RANGES")
progress("=" * 100)

for min_p, max_p in [(1.50, 5.00), (2.00, 5.00), (2.50, 5.00), (1.50, 4.00), (2.00, 4.00)]:
    for threshold in [0.70, 0.65, 0.60]:
        pf = df[(df['StartingPrice'] >= min_p) & (df['StartingPrice'] <= max_p)]
        t_df = pf[pf['WeightedScore'] >= threshold]
        if len(t_df) > 0:
            wins = t_df['ActualWinner'].sum()
            strike = wins / len(t_df) * 100
            avg_odds = t_df['StartingPrice'].mean()
            roi = ((wins * avg_odds) - len(t_df)) / len(t_df) * 100
            daily = len(t_df) / 7
            if roi > 20 and daily >= 2:
                progress(f"${min_p:.2f}-${max_p:.2f} @ {threshold:.2f}: {len(t_df):3} bets ({daily:.1f}/day) | Strike: {strike:.1f}% | ROI: {roi:+.1f}% <<<")
            else:
                progress(f"${min_p:.2f}-${max_p:.2f} @ {threshold:.2f}: {len(t_df):3} bets ({daily:.1f}/day) | Strike: {strike:.1f}% | ROI: {roi:+.1f}%")
