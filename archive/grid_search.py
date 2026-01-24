"""
Grid Search to find optimal parameters
Goal: 3-10 bets/day with ~30% ROI
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

progress("=" * 100)
progress("GRID SEARCH - Finding optimal parameters")
progress("=" * 100)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Load ALL data once for speed
progress("Loading all data (one-time)...\n")

# Get pace data before Nov 1 (use as cutoff for simplicity)
pace_df = pd.read_sql_query("""
WITH dog_pace_history_raw AS (
    SELECT 
        ge.GreyhoundID,
        rm.MeetingDate,
        (ge.FinishTimeBenchmarkLengths + rm.MeetingAvgBenchmarkLengths) as TotalBench,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.FinishTimeBenchmarkLengths IS NOT NULL
      AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
      AND rm.MeetingDate < '2025-11-01'
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

progress(f"Pace data: {len(pace_df):,} dogs")

# Get form data
form_df = pd.read_sql_query("""
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
      AND rm.MeetingDate < '2025-11-01'
)
SELECT 
    GreyhoundID,
    SUM(CASE WHEN RaceNum <= 5 AND IsWinner = 1 THEN 1 ELSE 0 END) as RawWins,
    COUNT(CASE WHEN RaceNum <= 5 THEN 1 END) as FormRaces
FROM dog_form_raw
GROUP BY GreyhoundID
""", conn)

progress(f"Form data: {len(form_df):,} dogs")

# Get November races (1 month test period)
races_df = pd.read_sql_query("""
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
  AND rm.MeetingDate >= '2025-11-01'
  AND rm.MeetingDate < '2025-12-01'
  AND t.TrackName NOT LIKE '%NZ%'
  AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
""", conn)

conn.close()

progress(f"November races: {len(races_df):,}")

# Merge
df = races_df.merge(pace_df, on='GreyhoundID', how='left')
df = df.merge(form_df, on='GreyhoundID', how='left')

# Clean
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df = df.dropna(subset=['StartingPrice', 'RawPaceAvg'])

progress(f"With pace data: {len(df):,}")

# Calculate base scores
pace_min = df['RawPaceAvg'].min()
pace_max = df['RawPaceAvg'].max()
df['PaceScore'] = (df['RawPaceAvg'] - pace_min) / (pace_max - pace_min)

df['FormRate'] = df.apply(
    lambda row: (row['RawWins'] / row['FormRaces'] * 100) if pd.notna(row['FormRaces']) and row['FormRaces'] > 0 else 0,
    axis=1
)
df['FormScore'] = df['FormRate'] / 100.0

# Days in November
days = 30

progress("\n" + "=" * 100)
progress("GRID SEARCH RESULTS")
progress("=" * 100)
progress(f"{'Config':<40} | {'Bets':>5} | {'Daily':>5} | {'Wins':>4} | {'Strike':>6} | {'ROI':>8}")
progress("-" * 100)

results = []

# Test different configurations
for pace_weight in [0.5, 0.6, 0.7, 0.8]:
    form_weight = 1.0 - pace_weight
    
    df['WeightedScore'] = (df['PaceScore'] * pace_weight) + (df['FormScore'] * form_weight)
    
    for threshold in [0.65, 0.70, 0.75, 0.80]:
        for price_min, price_max in [(1.50, 3.00), (1.50, 4.00), (2.00, 4.00), (2.50, 4.00), (2.00, 5.00)]:
            
            filtered = df[
                (df['WeightedScore'] >= threshold) & 
                (df['StartingPrice'] >= price_min) & 
                (df['StartingPrice'] <= price_max)
            ]
            
            if len(filtered) == 0:
                continue
            
            wins = filtered['ActualWinner'].sum()
            bets = len(filtered)
            strike = wins / bets * 100
            avg_odds = filtered['StartingPrice'].mean()
            profit = (wins * avg_odds) - bets
            roi = profit / bets * 100
            daily = bets / days
            
            config = f"P{pace_weight:.1f}/F{form_weight:.1f} @ {threshold:.2f} ${price_min:.2f}-${price_max:.2f}"
            
            results.append({
                'config': config,
                'pace_weight': pace_weight,
                'form_weight': form_weight,
                'threshold': threshold,
                'price_min': price_min,
                'price_max': price_max,
                'bets': bets,
                'daily': daily,
                'wins': wins,
                'strike': strike,
                'roi': roi
            })

# Sort by ROI and filter to 3-10 bets/day
df_results = pd.DataFrame(results)
good_volume = df_results[(df_results['daily'] >= 3) & (df_results['daily'] <= 15)]
good_volume = good_volume.sort_values('roi', ascending=False)

progress("\n" + "=" * 100)
progress("TOP CONFIGURATIONS (3-15 bets/day, sorted by ROI)")
progress("=" * 100)

for idx, row in good_volume.head(20).iterrows():
    marker = "<<<" if row['roi'] > 25 else ""
    progress(f"{row['config']:<40} | {row['bets']:5} | {row['daily']:5.1f} | {row['wins']:4} | {row['strike']:5.1f}% | {row['roi']:+7.1f}% {marker}")

# Also show best overall regardless of volume
progress("\n" + "=" * 100)
progress("BEST ROI CONFIGURATIONS (any volume)")
progress("=" * 100)

best_roi = df_results[df_results['bets'] >= 10].sort_values('roi', ascending=False)
for idx, row in best_roi.head(10).iterrows():
    progress(f"{row['config']:<40} | {row['bets']:5} | {row['daily']:5.1f} | {row['wins']:4} | {row['strike']:5.1f}% | {row['roi']:+7.1f}%")
