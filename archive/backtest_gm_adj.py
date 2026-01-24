"""
Backtest using G/M ADJ metrics (Split and OT)
These are the RELATIVE performance metrics that remove track condition effects

G/M OT ADJ = FinishTimeBenchmarkLengths - MeetingAvgBenchmarkLengths
G/M First Sec ADJ = SplitBenchmarkLengths - MeetingSplitAvgBenchmarkLengths

HIGHER = BETTER (faster relative to field that day)

This version processes WEEK BY WEEK to avoid look-ahead bias
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

progress("=" * 100)
progress("BACKTEST: G/M ADJ Metrics (Weekly Rolling, Last 5 Races)")
progress("=" * 100)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Test period: Sept-Nov 2025
TEST_START = '2025-09-01'
TEST_END = '2025-12-01'
LAST_N_RACES = 5  # Use last 5 races only

# Metro tracks get 4x weight, Provincial 2x, Country 1x
METRO_TRACKS = ['Wentworth Park', 'The Meadows', 'Sandown Park', 'Albion Park', 'Cannington']
PROVINCIAL_TRACKS = ['Richmond', 'Bulli', 'Dapto', 'Goulburn', 'Maitland', 'Gosford', 
                     'Geelong', 'Warragul', 'Ballarat', 'Warrnambool', 'Sale', 
                     'Ipswich', 'Bundaberg', 'Capalaba', 'Mandurah', 'Murray Bridge', 'Angle Park']

def get_track_tier(track_name):
    if track_name in METRO_TRACKS:
        return 4.0
    elif track_name in PROVINCIAL_TRACKS:
        return 2.0
    else:
        return 1.0

def get_dog_stats(conn, cutoff_date, last_n=5):
    """Get G/M ADJ averages for all dogs using races before cutoff_date"""
    query = f"""
    WITH ranked_races AS (
        SELECT 
            ge.GreyhoundID,
            rm.MeetingDate,
            t.TrackName,
            (ge.FinishTimeBenchmarkLengths - rm.MeetingAvgBenchmarkLengths) as GM_OT_ADJ,
            (ge.SplitBenchmarkLengths - rm.MeetingSplitAvgBenchmarkLengths) as GM_Split_ADJ,
            CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END as Winner,
            ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE ge.Position NOT IN ('DNF', 'SCR')
          AND ge.FinishTimeBenchmarkLengths IS NOT NULL
          AND rm.MeetingDate < '{cutoff_date}'
          AND t.TrackName NOT LIKE '%NZ%'
          AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
    )
    SELECT 
        GreyhoundID,
        TrackName,
        GM_OT_ADJ,
        GM_Split_ADJ,
        Winner,
        RaceNum
    FROM ranked_races
    WHERE RaceNum <= {last_n}
    """
    
    df = pd.read_sql_query(query, conn)
    if len(df) == 0:
        return pd.DataFrame()
    
    df['TrackWeight'] = df['TrackName'].apply(get_track_tier)
    
    # Calculate weighted averages per dog
    results = []
    for gid, group in df.groupby('GreyhoundID'):
        if len(group) < 3:  # Need at least 3 races
            continue
        
        weights = group['TrackWeight'].values
        
        # Weighted averages
        ot_avg = np.average(group['GM_OT_ADJ'].values, weights=weights)
        split_vals = group['GM_Split_ADJ'].fillna(0).values
        split_avg = np.average(split_vals, weights=weights)
        win_rate = group['Winner'].sum() / len(group) * 100
        
        results.append({
            'GreyhoundID': gid,
            'GM_OT_ADJ_Avg': ot_avg,
            'GM_Split_ADJ_Avg': split_avg,
            'RecentWinRate': win_rate,
            'RaceCount': len(group)
        })
    
    return pd.DataFrame(results)

# Get all test period races
progress("\nLoading test period races...")

test_query = f"""
SELECT 
    ge.GreyhoundID,
    ge.RaceID,
    ge.Box,
    ge.StartingPrice,
    ge.Position,
    r.RaceNumber,
    rm.MeetingDate,
    t.TrackName
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE ge.Position NOT IN ('DNF', 'SCR')
  AND ge.StartingPrice IS NOT NULL
  AND rm.MeetingDate >= '{TEST_START}'
  AND rm.MeetingDate < '{TEST_END}'
  AND t.TrackName NOT LIKE '%NZ%'
  AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
"""

test_df = pd.read_sql_query(test_query, conn)
test_df['StartingPrice'] = pd.to_numeric(test_df['StartingPrice'], errors='coerce')
test_df['Winner'] = (test_df['Position'] == '1').astype(int)
test_df['MeetingDate'] = pd.to_datetime(test_df['MeetingDate'])

progress(f"Loaded {len(test_df):,} test entries")

# Get unique weeks
test_df['Week'] = test_df['MeetingDate'].dt.to_period('W').dt.start_time
weeks = sorted(test_df['Week'].unique())
progress(f"Processing {len(weeks)} weeks...")

# Process week by week
all_results = []

for week_start in weeks:
    week_end = week_start + timedelta(days=7)
    week_races = test_df[(test_df['MeetingDate'] >= week_start) & (test_df['MeetingDate'] < week_end)].copy()
    
    if len(week_races) == 0:
        continue
    
    # Get dog stats using data BEFORE this week
    cutoff = week_start.strftime('%Y-%m-%d')
    dog_stats = get_dog_stats(conn, cutoff, LAST_N_RACES)
    
    if len(dog_stats) == 0:
        continue
    
    # Merge
    week_races = week_races.merge(dog_stats, on='GreyhoundID', how='inner')
    
    if len(week_races) > 0:
        all_results.append(week_races)

# Combine all weeks
test_df = pd.concat(all_results, ignore_index=True)
progress(f"After weekly processing: {len(test_df):,} entries with G/M ADJ data")

# For each race, rank dogs by their GM_OT_ADJ_Avg (higher = better)
test_df['RacePaceRank'] = test_df.groupby('RaceID')['GM_OT_ADJ_Avg'].rank(ascending=False, method='first')

# Also rank by split time
test_df['RaceSplitRank'] = test_df.groupby('RaceID')['GM_Split_ADJ_Avg'].rank(ascending=False, method='first')

# Mark favourites
test_df['IsFavourite'] = test_df.groupby('RaceID')['StartingPrice'].transform(lambda x: x == x.min())

def calc_roi(subset):
    if len(subset) == 0:
        return 0, 0, 0
    wins = subset['Winner'].sum()
    bets = len(subset)
    strike = wins / bets * 100
    profit = (subset['Winner'] * subset['StartingPrice']).sum() - bets
    roi = profit / bets * 100
    return bets, strike, roi

progress("\n" + "=" * 100)
progress("ANALYSIS 1: Best GM_OT_ADJ (highest pace advantage)")
progress("=" * 100)

for rank in [1, 2, 3]:
    subset = test_df[test_df['RacePaceRank'] == rank]
    bets, strike, roi = calc_roi(subset)
    avg_odds = subset['StartingPrice'].mean()
    progress(f"Best GM_OT_ADJ rank {rank}: {bets:,} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 2: Best GM_Split_ADJ (fastest early speed)")
progress("=" * 100)

for rank in [1, 2, 3]:
    subset = test_df[test_df['RaceSplitRank'] == rank]
    bets, strike, roi = calc_roi(subset)
    avg_odds = subset['StartingPrice'].mean()
    progress(f"Best GM_Split_ADJ rank {rank}: {bets:,} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 3: Combining metrics")
progress("=" * 100)

# Best OT + Best Split (same dog is best at both)
combo = test_df[(test_df['RacePaceRank'] == 1) & (test_df['RaceSplitRank'] == 1)]
bets, strike, roi = calc_roi(combo)
avg_odds = combo['StartingPrice'].mean()
progress(f"Best OT + Best Split: {bets:,} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

# Best OT + Top 2 Split
combo = test_df[(test_df['RacePaceRank'] == 1) & (test_df['RaceSplitRank'] <= 2)]
bets, strike, roi = calc_roi(combo)
avg_odds = combo['StartingPrice'].mean()
progress(f"Best OT + Top-2 Split: {bets:,} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

# Top 2 OT + Best Split
combo = test_df[(test_df['RacePaceRank'] <= 2) & (test_df['RaceSplitRank'] == 1)]
bets, strike, roi = calc_roi(combo)
avg_odds = combo['StartingPrice'].mean()
progress(f"Top-2 OT + Best Split: {bets:,} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 4: With market filters")
progress("=" * 100)

# Best GM_OT_ADJ + favourite
subset = test_df[(test_df['RacePaceRank'] == 1) & (test_df['IsFavourite'])]
bets, strike, roi = calc_roi(subset)
avg_odds = subset['StartingPrice'].mean()
progress(f"Best GM_OT_ADJ + Favourite: {bets} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

# Best GM_OT_ADJ but NOT favourite (value)
subset = test_df[(test_df['RacePaceRank'] == 1) & (~test_df['IsFavourite'])]
bets, strike, roi = calc_roi(subset)
avg_odds = subset['StartingPrice'].mean()
progress(f"Best GM_OT_ADJ NOT Favourite: {bets} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

# Best GM_OT_ADJ + Box 1
for box in [1, 2, 8]:
    subset = test_df[(test_df['RacePaceRank'] == 1) & (test_df['Box'] == box)]
    bets, strike, roi = calc_roi(subset)
    avg_odds = subset['StartingPrice'].mean()
    progress(f"Best GM_OT_ADJ + Box {box}: {bets} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 5: Price range filters")
progress("=" * 100)

for low, high in [(1.5, 3.0), (2.0, 4.0), (2.5, 5.0), (3.0, 6.0)]:
    subset = test_df[(test_df['RacePaceRank'] == 1) & 
                     (test_df['StartingPrice'] >= low) & 
                     (test_df['StartingPrice'] < high)]
    bets, strike, roi = calc_roi(subset)
    progress(f"Best GM_OT_ADJ ${low}-${high}: {bets} bets | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 6: Combining everything")
progress("=" * 100)

results = []

configs = [
    ("Best OT + Fav", (test_df['RacePaceRank'] == 1) & (test_df['IsFavourite'])),
    ("Best OT + Best Split + Fav", (test_df['RacePaceRank'] == 1) & (test_df['RaceSplitRank'] == 1) & (test_df['IsFavourite'])),
    ("Best OT + NOT Fav + $2-5", (test_df['RacePaceRank'] == 1) & (~test_df['IsFavourite']) & (test_df['StartingPrice'] >= 2) & (test_df['StartingPrice'] < 5)),
    ("Best OT + Box 1", (test_df['RacePaceRank'] == 1) & (test_df['Box'] == 1)),
    ("Best OT + Box 1 + $2-5", (test_df['RacePaceRank'] == 1) & (test_df['Box'] == 1) & (test_df['StartingPrice'] >= 2) & (test_df['StartingPrice'] < 5)),
    ("Best Split + Box 1", (test_df['RaceSplitRank'] == 1) & (test_df['Box'] == 1)),
    ("Best Split + Fav", (test_df['RaceSplitRank'] == 1) & (test_df['IsFavourite'])),
    ("Best Split + NOT Fav + $2-5", (test_df['RaceSplitRank'] == 1) & (~test_df['IsFavourite']) & (test_df['StartingPrice'] >= 2) & (test_df['StartingPrice'] < 5)),
    ("Best OT + Best Split + Box 1", (test_df['RacePaceRank'] == 1) & (test_df['RaceSplitRank'] == 1) & (test_df['Box'] == 1)),
    ("Top-2 OT + Top-2 Split + Fav", (test_df['RacePaceRank'] <= 2) & (test_df['RaceSplitRank'] <= 2) & (test_df['IsFavourite'])),
    ("Best OT + $1.5-3", (test_df['RacePaceRank'] == 1) & (test_df['StartingPrice'] >= 1.5) & (test_df['StartingPrice'] < 3)),
    ("Best OT + $2-4", (test_df['RacePaceRank'] == 1) & (test_df['StartingPrice'] >= 2) & (test_df['StartingPrice'] < 4)),
    ("Best OT + $2.5-5", (test_df['RacePaceRank'] == 1) & (test_df['StartingPrice'] >= 2.5) & (test_df['StartingPrice'] < 5)),
]

for name, mask in configs:
    subset = test_df[mask]
    bets, strike, roi = calc_roi(subset)
    if bets >= 30:
        daily = bets / 91
        results.append((name, bets, daily, strike, roi))

results.sort(key=lambda x: x[4], reverse=True)

progress("\nAll combinations sorted by ROI:")
progress("-" * 90)
for name, bets, daily, strike, roi in results:
    progress(f"{name:45s} | {bets:4d} ({daily:5.1f}/day) | {strike:5.1f}% | ROI {roi:+6.1f}%")

conn.close()

progress("\n" + "=" * 100)
progress("SUMMARY")
progress("=" * 100)
progress("""
The G/M ADJ metrics measure how fast a dog ran RELATIVE to the meeting average.
This removes track condition effects (wet tracks, headwinds, etc.)

Key finding: Using RELATIVE pace within each race rather than absolute values
may be the key to finding edge.
""")
