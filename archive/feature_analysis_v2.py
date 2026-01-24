"""
Feature Analysis V2 - Focus on finding profitable edges
Key insight: Need to find situations where actual win rate > implied probability
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
progress("FEATURE ANALYSIS V2 - Finding profitable edges")
progress("=" * 100)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Load full data with historical pace for each dog
progress("\nLoading race data with in-race features...")

query = """
SELECT 
    ge.GreyhoundID,
    ge.Box,
    ge.StartingPrice,
    ge.Position,
    ge.FinishTimeBenchmarkLengths,
    ge.SplitBenchmarkLengths,
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
  AND rm.MeetingDate >= '2025-09-01'
  AND rm.MeetingDate < '2025-12-01'
  AND t.TrackName NOT LIKE '%NZ%'
  AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
"""

df = pd.read_sql_query(query, conn)
progress(f"Loaded {len(df):,} entries across 3 months")

# Convert numeric columns
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['Box'] = pd.to_numeric(df['Box'], errors='coerce')
df['ImpliedProb'] = 1.0 / df['StartingPrice']

def calc_roi(subset):
    """Calculate ROI for a subset of bets"""
    if len(subset) == 0:
        return 0, 0, 0
    wins = subset['Winner'].sum()
    bets = len(subset)
    strike = wins / bets * 100
    profit = (subset['Winner'] * subset['StartingPrice']).sum() - bets
    roi = profit / bets * 100
    return bets, strike, roi

progress("\n" + "=" * 100)
progress("ANALYSIS 1: BOX NUMBER ROI (betting on specific boxes)")
progress("=" * 100)

for box in range(1, 9):
    subset = df[df['Box'] == box]
    bets, strike, roi = calc_roi(subset)
    avg_odds = subset['StartingPrice'].mean()
    progress(f"Box {box}: {bets:,} bets | Avg odds ${avg_odds:.2f} | Strike {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 2: BOX + FAVOURITE COMBO")
progress("=" * 100)

# Mark favourites (lowest odds in race)
df['IsFavourite'] = df.groupby('RaceID')['StartingPrice'].transform(lambda x: x == x.min())

# Box 1 favourite
for box in range(1, 9):
    subset = df[(df['Box'] == box) & (df['IsFavourite'])]
    bets, strike, roi = calc_roi(subset)
    avg_odds = subset['StartingPrice'].mean()
    progress(f"Box {box} Favourite: {bets} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 3: RACE NUMBER + FAVOURITE")
progress("=" * 100)

for race_num in range(1, 13):
    subset = df[(df['RaceNumber'] == race_num) & (df['IsFavourite'])]
    bets, strike, roi = calc_roi(subset)
    avg_odds = subset['StartingPrice'].mean()
    progress(f"Race {race_num:2d} Fav: {bets} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 4: SECOND FAVOURITE (better value?)")
progress("=" * 100)

# Mark 2nd favourite
df['Rank'] = df.groupby('RaceID')['StartingPrice'].rank(method='first')
df['Is2ndFav'] = df['Rank'] == 2

second_fav = df[df['Is2ndFav']]
bets, strike, roi = calc_roi(second_fav)
avg_odds = second_fav['StartingPrice'].mean()
progress(f"All 2nd Favourites: {bets} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

# 2nd fav in specific price ranges
progress("\n2nd Favourite by price range:")
for low, high in [(2.5, 4.0), (3.0, 5.0), (4.0, 6.0), (5.0, 8.0)]:
    subset = second_fav[(second_fav['StartingPrice'] >= low) & (second_fav['StartingPrice'] < high)]
    if len(subset) < 50:
        continue
    bets, strike, roi = calc_roi(subset)
    progress(f"  ${low:.1f}-${high:.1f}: {bets} bets | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 5: HEAVILY BACKED (price dropped vs implied)")
progress("=" * 100)

# Can't measure price movement directly, but short-priced favourites might indicate confidence
short_fav = df[(df['IsFavourite']) & (df['StartingPrice'] <= 1.80)]
bets, strike, roi = calc_roi(short_fav)
progress(f"Short-priced favourites (<=$1.80): {bets} bets | {strike:.1f}% | ROI {roi:+.1f}%")

medium_fav = df[(df['IsFavourite']) & (df['StartingPrice'] > 1.80) & (df['StartingPrice'] <= 2.50)]
bets, strike, roi = calc_roi(medium_fav)
progress(f"Medium favourites ($1.81-$2.50): {bets} bets | {strike:.1f}% | ROI {roi:+.1f}%")

longer_fav = df[(df['IsFavourite']) & (df['StartingPrice'] > 2.50) & (df['StartingPrice'] <= 4.00)]
bets, strike, roi = calc_roi(longer_fav)
progress(f"Longer favourites ($2.51-$4.00): {bets} bets | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 6: INRUN POSITION (led at first split)")
progress("=" * 100)

# InRun format appears to be position at each split point
# First digit = position at first split
df['InRunFirst'] = df['InRun'].astype(str).str[0]

for pos in ['1', '2', '3']:
    subset = df[df['InRunFirst'] == pos]
    bets, strike, roi = calc_roi(subset)
    progress(f"Led at 1st split (position {pos}): {bets:,} bets | {strike:.1f}% strike | ROI {roi:+.1f}%")

# Dogs that led AND were favourite
leader_fav = df[(df['InRunFirst'] == '1') & (df['IsFavourite'])]
bets, strike, roi = calc_roi(leader_fav)
avg_odds = leader_fav['StartingPrice'].mean()
progress(f"\nFavourites that led at 1st split: {bets} bets | {avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 7: RELATIVE PACE IN RACE (best pace vs field)")
progress("=" * 100)

# Calculate relative pace within each race
# First get historical pace for each dog

pace_query = """
WITH dog_pace_history AS (
    SELECT 
        ge.GreyhoundID,
        rm.MeetingDate,
        (ge.FinishTimeBenchmarkLengths + rm.MeetingAvgBenchmarkLengths) as TotalBench,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE ge.Position NOT IN ('DNF', 'SCR')
      AND ge.FinishTimeBenchmarkLengths IS NOT NULL
      AND rm.MeetingDate < '2025-12-01'
),
dog_pace_avg AS (
    SELECT 
        GreyhoundID,
        AVG(CASE WHEN RaceNum <= 5 THEN TotalBench END) as HistoricalPace,
        COUNT(CASE WHEN RaceNum <= 5 THEN 1 END) as RaceCount
    FROM dog_pace_history
    GROUP BY GreyhoundID
    HAVING RaceCount >= 3
)
SELECT GreyhoundID, HistoricalPace, RaceCount
FROM dog_pace_avg
"""

progress("Loading historical pace data...")
pace_df = pd.read_sql_query(pace_query, conn)
progress(f"Got pace data for {len(pace_df):,} dogs")

# Merge pace into main data
df = df.merge(pace_df, on='GreyhoundID', how='left')

# For each race, rank dogs by pace (higher = better)
df['RacePaceRank'] = df.groupby('RaceID')['HistoricalPace'].rank(ascending=False, method='first')

progress("\nPerformance by pace rank within race:")
for rank in [1, 2, 3]:
    subset = df[df['RacePaceRank'] == rank]
    bets, strike, roi = calc_roi(subset)
    avg_odds = subset['StartingPrice'].mean()
    progress(f"Best pace rank {rank}: {bets:,} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

# Best pace AND favourite
best_pace_fav = df[(df['RacePaceRank'] == 1) & (df['IsFavourite'])]
bets, strike, roi = calc_roi(best_pace_fav)
avg_odds = best_pace_fav['StartingPrice'].mean()
progress(f"\nBest pace + Favourite: {bets} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

# Best pace but NOT favourite (value?)
best_pace_not_fav = df[(df['RacePaceRank'] == 1) & (~df['IsFavourite'])]
bets, strike, roi = calc_roi(best_pace_not_fav)
avg_odds = best_pace_not_fav['StartingPrice'].mean()
progress(f"Best pace NOT favourite: {bets} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 8: COMBINING FACTORS")
progress("=" * 100)

# Best pace + Box 1
combo = df[(df['RacePaceRank'] == 1) & (df['Box'] == 1)]
bets, strike, roi = calc_roi(combo)
avg_odds = combo['StartingPrice'].mean()
progress(f"Best pace + Box 1: {bets} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

# Best pace + Box 1 + specific price range
for low, high in [(1.5, 3.0), (2.0, 4.0), (2.5, 5.0)]:
    combo = df[(df['RacePaceRank'] == 1) & (df['Box'] == 1) & 
               (df['StartingPrice'] >= low) & (df['StartingPrice'] < high)]
    if len(combo) < 20:
        continue
    bets, strike, roi = calc_roi(combo)
    progress(f"Best pace + Box 1 + ${low}-${high}: {bets} bets | {strike:.1f}% | ROI {roi:+.1f}%")

# Top 2 pace + Favourite + good box
progress("\nTop-2 pace + Favourite + Boxes 1,2,8:")
combo = df[(df['RacePaceRank'] <= 2) & (df['IsFavourite']) & (df['Box'].isin([1, 2, 8]))]
bets, strike, roi = calc_roi(combo)
avg_odds = combo['StartingPrice'].mean()
progress(f"Result: {bets} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

# By price range
for low, high in [(1.5, 2.5), (2.0, 3.5), (2.5, 4.0)]:
    combo = df[(df['RacePaceRank'] <= 2) & (df['IsFavourite']) & (df['Box'].isin([1, 2, 8])) &
               (df['StartingPrice'] >= low) & (df['StartingPrice'] < high)]
    if len(combo) < 30:
        continue
    bets, strike, roi = calc_roi(combo)
    progress(f"  + ${low}-${high}: {bets} bets | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 9: DISTANCE SPECIALISTS")
progress("=" * 100)

# Group by distance
for dist in [300, 400, 500, 600]:
    subset = df[(df['Distance'] >= dist - 50) & (df['Distance'] < dist + 50) & (df['IsFavourite'])]
    if len(subset) < 50:
        continue
    bets, strike, roi = calc_roi(subset)
    progress(f"{dist}m races (favourites): {bets} bets | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("SUMMARY: BEST PERFORMING COMBINATIONS")
progress("=" * 100)

# Run all promising combinations
results = []

# Test various filters
configs = [
    # (name, filter_func)
    ("Favourites (all)", lambda x: x['IsFavourite']),
    ("Fav + Box 1", lambda x: (x['IsFavourite']) & (x['Box'] == 1)),
    ("Fav + Box 2", lambda x: (x['IsFavourite']) & (x['Box'] == 2)),
    ("Fav + Box 8", lambda x: (x['IsFavourite']) & (x['Box'] == 8)),
    ("Fav + Race 1", lambda x: (x['IsFavourite']) & (x['RaceNumber'] == 1)),
    ("Fav + Race 5", lambda x: (x['IsFavourite']) & (x['RaceNumber'] == 5)),
    ("Fav + Race 1 or 5", lambda x: (x['IsFavourite']) & (x['RaceNumber'].isin([1, 5]))),
    ("Best Pace + Fav", lambda x: (x['RacePaceRank'] == 1) & (x['IsFavourite'])),
    ("Best Pace + Box 1", lambda x: (x['RacePaceRank'] == 1) & (x['Box'] == 1)),
    ("Best Pace + Fav + Box 1", lambda x: (x['RacePaceRank'] == 1) & (x['IsFavourite']) & (x['Box'] == 1)),
    ("Best Pace + Fav + Race 1", lambda x: (x['RacePaceRank'] == 1) & (x['IsFavourite']) & (x['RaceNumber'] == 1)),
    ("Fav + Box 1/2/8 + $1.5-3", lambda x: (x['IsFavourite']) & (x['Box'].isin([1, 2, 8])) & (x['StartingPrice'] >= 1.5) & (x['StartingPrice'] < 3)),
    ("2nd Fav + $3-6", lambda x: (x['Is2ndFav']) & (x['StartingPrice'] >= 3) & (x['StartingPrice'] < 6)),
    ("Short Fav (<$1.80)", lambda x: (x['IsFavourite']) & (x['StartingPrice'] <= 1.80)),
    ("Fav $1.80-$2.50", lambda x: (x['IsFavourite']) & (x['StartingPrice'] > 1.80) & (x['StartingPrice'] <= 2.50)),
]

for name, filter_func in configs:
    try:
        mask = filter_func(df)
        subset = df[mask]
        bets, strike, roi = calc_roi(subset)
        if bets >= 50:
            daily_bets = bets / 91  # 3 months
            results.append((name, bets, daily_bets, strike, roi))
    except Exception as e:
        progress(f"Error with {name}: {e}")

# Sort by ROI
results.sort(key=lambda x: x[4], reverse=True)

progress("\nAll combinations sorted by ROI (minimum 50 bets):")
progress("-" * 80)
for name, bets, daily, strike, roi in results:
    progress(f"{name:40s} | {bets:4d} ({daily:.1f}/day) | {strike:5.1f}% | ROI {roi:+6.1f}%")

conn.close()

progress("\n" + "=" * 100)
progress("CONCLUSION")
progress("=" * 100)
progress("""
The market is very efficient. Key insights:
1. Short-priced favourites ($<1.80) lose less than longer favourites
2. Race 1 and Race 5 favourites show slightly better performance
3. Box position alone doesn't create betting edge
4. Best historical pace doesn't beat the market when bet blindly
5. Need to find MISPRICED situations, not just winners
""")
