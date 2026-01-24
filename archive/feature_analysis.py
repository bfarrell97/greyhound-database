"""
Feature Analysis - What actually predicts winners?
Look at all available data to find predictive signals
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

progress("=" * 100)
progress("FEATURE ANALYSIS - Finding predictive signals")
progress("=" * 100)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# First, let's see what columns we have available
progress("\nChecking available data...", indent=1)

# Check GreyhoundEntries schema
schema = pd.read_sql_query("PRAGMA table_info(GreyhoundEntries)", conn)
progress("\nGreyhoundEntries columns:", indent=1)
for _, row in schema.iterrows():
    progress(f"  {row['name']}: {row['type']}", indent=2)

# Check what data we have for November races
progress("\n" + "=" * 100)
progress("Loading November race data with ALL features...")

query = """
SELECT 
    ge.*,
    g.GreyhoundName,
    r.RaceID,
    r.Distance,
    r.RaceNumber,
    rm.MeetingDate,
    t.TrackName
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
LIMIT 100
"""

sample = pd.read_sql_query(query, conn)
progress(f"\nSample columns: {list(sample.columns)}")
progress(f"\nSample data (first 3 rows):")
print(sample.head(3).T)

# Now get full data for analysis
progress("\n" + "=" * 100)
progress("Loading full November data for feature analysis...")

query = """
SELECT 
    ge.GreyhoundID,
    ge.Box,
    ge.StartingPrice,
    ge.Position,
    ge.FinishTimeBenchmarkLengths,
    ge.SplitBenchmarkLengths,
    ge.EarlySpeed,
    ge.Rating,
    ge.InRun,
    r.RaceID,
    r.Distance,
    r.RaceNumber,
    rm.MeetingDate,
    rm.MeetingAvgBenchmarkLengths,
    t.TrackName,
    (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as Winner
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
"""

df = pd.read_sql_query(query, conn)
progress(f"Loaded {len(df):,} entries")

# Convert numeric columns
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['Box'] = pd.to_numeric(df['Box'], errors='coerce')

# Calculate implied probability
df['ImpliedProb'] = 1.0 / df['StartingPrice']

progress("\n" + "=" * 100)
progress("FACTOR 1: BOX NUMBER")
progress("=" * 100)

box_stats = df.groupby('Box').agg({
    'Winner': ['sum', 'count', 'mean'],
    'StartingPrice': 'mean'
}).round(3)
box_stats.columns = ['Wins', 'Races', 'WinRate', 'AvgOdds']
box_stats['ImpliedWinRate'] = 1 / box_stats['AvgOdds']
box_stats['EdgeVsMarket'] = box_stats['WinRate'] - box_stats['ImpliedWinRate']

progress("\nWin rate by box number:")
print(box_stats)

# Best boxes
progress("\nBoxes with positive edge vs market:")
positive_edge = box_stats[box_stats['EdgeVsMarket'] > 0]
print(positive_edge)

progress("\n" + "=" * 100)
progress("FACTOR 2: STARTING PRICE ACCURACY")
progress("=" * 100)

# Group by odds bracket
df['OddsBracket'] = pd.cut(df['StartingPrice'], 
                            bins=[0, 2, 3, 4, 5, 7, 10, 100],
                            labels=['$1-2', '$2-3', '$3-4', '$4-5', '$5-7', '$7-10', '$10+'])

odds_stats = df.groupby('OddsBracket').agg({
    'Winner': ['sum', 'count', 'mean'],
    'ImpliedProb': 'mean'
}).round(3)
odds_stats.columns = ['Wins', 'Races', 'ActualWinRate', 'ImpliedWinRate']
odds_stats['EdgeVsMarket'] = odds_stats['ActualWinRate'] - odds_stats['ImpliedWinRate']

progress("\nActual vs Implied win rate by odds bracket:")
print(odds_stats)

progress("\n" + "=" * 100)
progress("FACTOR 3: TRACK-SPECIFIC PATTERNS")
progress("=" * 100)

track_stats = df.groupby('TrackName').agg({
    'Winner': ['sum', 'count', 'mean'],
    'ImpliedProb': 'mean'
}).round(3)
track_stats.columns = ['Wins', 'Races', 'WinRate', 'AvgImpliedProb']

# Only tracks with enough data
track_stats = track_stats[track_stats['Races'] >= 100]
track_stats['EdgeVsMarket'] = track_stats['WinRate'] - track_stats['AvgImpliedProb']
track_stats = track_stats.sort_values('EdgeVsMarket', ascending=False)

progress("\nTracks with edge (vs average market implied prob):")
print(track_stats.head(10))

progress("\n" + "=" * 100)
progress("FACTOR 4: FAVOURITE WIN RATE")
progress("=" * 100)

# For each race, identify favourite
df_with_fav = df.copy()
df_with_fav['IsFavourite'] = df_with_fav.groupby('RaceID')['StartingPrice'].transform(lambda x: x == x.min())

fav_stats = df_with_fav[df_with_fav['IsFavourite']].agg({
    'Winner': ['sum', 'count', 'mean'],
    'ImpliedProb': 'mean',
    'StartingPrice': 'mean'
})

progress(f"\nFavourites win rate: {fav_stats['Winner']['mean']*100:.1f}%")
progress(f"Market implied: {fav_stats['ImpliedProb']['mean']*100:.1f}%")
progress(f"Edge: {(fav_stats['Winner']['mean'] - fav_stats['ImpliedProb']['mean'])*100:.1f}%")
progress(f"Avg favourite odds: ${fav_stats['StartingPrice']['mean']:.2f}")

# ROI from backing all favourites
fav_df = df_with_fav[df_with_fav['IsFavourite']]
fav_profit = (fav_df['Winner'] * fav_df['StartingPrice']).sum() - len(fav_df)
fav_roi = fav_profit / len(fav_df) * 100
progress(f"ROI backing all favourites: {fav_roi:+.1f}%")

progress("\n" + "=" * 100)
progress("FACTOR 5: FAVOURITE + BOX COMBO")
progress("=" * 100)

# Favourites from good boxes
for box in [1, 2, 5, 8]:
    fav_box = df_with_fav[(df_with_fav['IsFavourite']) & (df_with_fav['Box'] == box)]
    if len(fav_box) < 20:
        continue
    wins = fav_box['Winner'].sum()
    strike = wins / len(fav_box) * 100
    avg_odds = fav_box['StartingPrice'].mean()
    profit = (fav_box['Winner'] * fav_box['StartingPrice']).sum() - len(fav_box)
    roi = profit / len(fav_box) * 100
    progress(f"Favourites from Box {box}: {len(fav_box)} bets | {strike:.1f}% strike | ROI: {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("FACTOR 6: RACE NUMBER (early vs late)")
progress("=" * 100)

race_num_stats = df_with_fav[df_with_fav['IsFavourite']].groupby('RaceNumber').agg({
    'Winner': ['sum', 'count', 'mean'],
    'StartingPrice': 'mean'
})
race_num_stats.columns = ['Wins', 'Races', 'WinRate', 'AvgOdds']
race_num_stats['ROI'] = (race_num_stats['Wins'] * race_num_stats['AvgOdds'] - race_num_stats['Races']) / race_num_stats['Races'] * 100

progress("\nFavourite performance by race number:")
print(race_num_stats)

conn.close()

progress("\n" + "=" * 100)
progress("SUMMARY - Potential strategies")
progress("=" * 100)
progress("""
Based on analysis:
1. Box numbers may have small edges (check which boxes)
2. Favourites win at roughly market rate (no edge)
3. Need to find COMBINATIONS that create edge

Next steps:
- Test favourite + specific box + price range combos
- Look at historical pace/form RELATIVE to race field
- Consider sectional times if available
""")
