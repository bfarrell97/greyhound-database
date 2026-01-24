"""
Compare two pace calculations:
1. ADDITIVE: FinishTimeBenchmarkLengths + MeetingAvgBenchmarkLengths (original)
2. RELATIVE: FinishTimeBenchmarkLengths - MeetingAvgBenchmarkLengths (G/M ADJ)

Test which one is actually predictive when properly backtested
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
progress("COMPARISON: Additive vs Relative Pace Metrics")
progress("=" * 100)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

TEST_START = '2025-09-01'
TEST_END = '2025-12-01'
LAST_N_RACES = 5

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
    """Get both additive and relative pace metrics"""
    query = f"""
    WITH ranked_races AS (
        SELECT 
            ge.GreyhoundID,
            rm.MeetingDate,
            t.TrackName,
            -- ADDITIVE (original formula)
            (ge.FinishTimeBenchmarkLengths + rm.MeetingAvgBenchmarkLengths) as Pace_Add,
            (ge.SplitBenchmarkLengths + rm.MeetingSplitAvgBenchmarkLengths) as Split_Add,
            -- RELATIVE (G/M ADJ formula)
            (ge.FinishTimeBenchmarkLengths - rm.MeetingAvgBenchmarkLengths) as Pace_Rel,
            (ge.SplitBenchmarkLengths - rm.MeetingSplitAvgBenchmarkLengths) as Split_Rel,
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
        Pace_Add,
        Split_Add,
        Pace_Rel,
        Split_Rel,
        Winner,
        RaceNum
    FROM ranked_races
    WHERE RaceNum <= {last_n}
    """
    
    df = pd.read_sql_query(query, conn)
    if len(df) == 0:
        return pd.DataFrame()
    
    df['TrackWeight'] = df['TrackName'].apply(get_track_tier)
    
    results = []
    for gid, group in df.groupby('GreyhoundID'):
        if len(group) < 3:
            continue
        
        weights = group['TrackWeight'].values
        
        # Additive metrics (weighted avg)
        pace_add = np.average(group['Pace_Add'].values, weights=weights)
        split_add = np.average(group['Split_Add'].fillna(0).values, weights=weights)
        
        # Relative metrics (weighted avg)
        pace_rel = np.average(group['Pace_Rel'].values, weights=weights)
        split_rel = np.average(group['Split_Rel'].fillna(0).values, weights=weights)
        
        win_rate = group['Winner'].sum() / len(group) * 100
        
        results.append({
            'GreyhoundID': gid,
            'Pace_Add_Avg': pace_add,
            'Split_Add_Avg': split_add,
            'Pace_Rel_Avg': pace_rel,
            'Split_Rel_Avg': split_rel,
            'WinRate': win_rate,
            'RaceCount': len(group)
        })
    
    return pd.DataFrame(results)

# Load test data
progress("\nLoading test period races...")

test_query = f"""
SELECT 
    ge.GreyhoundID,
    ge.RaceID,
    ge.Box,
    ge.StartingPrice,
    ge.Position,
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

# Weekly processing
test_df['Week'] = test_df['MeetingDate'].dt.to_period('W').dt.start_time
weeks = sorted(test_df['Week'].unique())
progress(f"Processing {len(weeks)} weeks...")

all_results = []
for week_start in weeks:
    week_end = week_start + timedelta(days=7)
    week_races = test_df[(test_df['MeetingDate'] >= week_start) & (test_df['MeetingDate'] < week_end)].copy()
    
    if len(week_races) == 0:
        continue
    
    cutoff = week_start.strftime('%Y-%m-%d')
    dog_stats = get_dog_stats(conn, cutoff, LAST_N_RACES)
    
    if len(dog_stats) == 0:
        continue
    
    week_races = week_races.merge(dog_stats, on='GreyhoundID', how='inner')
    
    if len(week_races) > 0:
        all_results.append(week_races)

test_df = pd.concat(all_results, ignore_index=True)
progress(f"After processing: {len(test_df):,} entries")

# Rank by each metric within each race
# ADDITIVE: Higher = better (faster overall)
test_df['Rank_Pace_Add'] = test_df.groupby('RaceID')['Pace_Add_Avg'].rank(ascending=False, method='first')
test_df['Rank_Split_Add'] = test_df.groupby('RaceID')['Split_Add_Avg'].rank(ascending=False, method='first')

# RELATIVE: Higher = better (faster relative to meeting)
test_df['Rank_Pace_Rel'] = test_df.groupby('RaceID')['Pace_Rel_Avg'].rank(ascending=False, method='first')
test_df['Rank_Split_Rel'] = test_df.groupby('RaceID')['Split_Rel_Avg'].rank(ascending=False, method='first')

# Favourite
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
progress("ADDITIVE FORMULA (FinishTimeBenchmark + MeetingAvgBenchmark)")
progress("=" * 100)

for rank in [1, 2, 3]:
    subset = test_df[test_df['Rank_Pace_Add'] == rank]
    bets, strike, roi = calc_roi(subset)
    avg_odds = subset['StartingPrice'].mean()
    progress(f"Rank {rank} (Additive OT): {bets:,} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

subset = test_df[(test_df['Rank_Pace_Add'] == 1) & (test_df['IsFavourite'])]
bets, strike, roi = calc_roi(subset)
progress(f"Best Additive OT + Fav: {bets} bets | {strike:.1f}% | ROI {roi:+.1f}%")

subset = test_df[(test_df['Rank_Pace_Add'] == 1) & (~test_df['IsFavourite'])]
bets, strike, roi = calc_roi(subset)
avg_odds = subset['StartingPrice'].mean()
progress(f"Best Additive OT NOT Fav: {bets} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("RELATIVE FORMULA (FinishTimeBenchmark - MeetingAvgBenchmark)")
progress("=" * 100)

for rank in [1, 2, 3]:
    subset = test_df[test_df['Rank_Pace_Rel'] == rank]
    bets, strike, roi = calc_roi(subset)
    avg_odds = subset['StartingPrice'].mean()
    progress(f"Rank {rank} (Relative OT): {bets:,} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

subset = test_df[(test_df['Rank_Pace_Rel'] == 1) & (test_df['IsFavourite'])]
bets, strike, roi = calc_roi(subset)
progress(f"Best Relative OT + Fav: {bets} bets | {strike:.1f}% | ROI {roi:+.1f}%")

subset = test_df[(test_df['Rank_Pace_Rel'] == 1) & (~test_df['IsFavourite'])]
bets, strike, roi = calc_roi(subset)
avg_odds = subset['StartingPrice'].mean()
progress(f"Best Relative OT NOT Fav: {bets} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("SPLIT TIME COMPARISON")
progress("=" * 100)

subset = test_df[(test_df['Rank_Split_Add'] == 1) & (test_df['IsFavourite'])]
bets, strike, roi = calc_roi(subset)
progress(f"Best Additive Split + Fav: {bets} bets | {strike:.1f}% | ROI {roi:+.1f}%")

subset = test_df[(test_df['Rank_Split_Rel'] == 1) & (test_df['IsFavourite'])]
bets, strike, roi = calc_roi(subset)
progress(f"Best Relative Split + Fav: {bets} bets | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("PRICE RANGE BREAKDOWN")
progress("=" * 100)

for low, high in [(1.5, 2.5), (2.0, 3.5), (2.5, 4.0), (3.0, 5.0)]:
    # Additive
    subset = test_df[(test_df['Rank_Pace_Add'] == 1) & 
                     (test_df['StartingPrice'] >= low) & 
                     (test_df['StartingPrice'] < high)]
    bets, strike, roi = calc_roi(subset)
    progress(f"Add OT ${low}-${high}: {bets} bets | {strike:.1f}% | ROI {roi:+.1f}%")
    
    # Relative
    subset = test_df[(test_df['Rank_Pace_Rel'] == 1) & 
                     (test_df['StartingPrice'] >= low) & 
                     (test_df['StartingPrice'] < high)]
    bets, strike, roi = calc_roi(subset)
    progress(f"Rel OT ${low}-${high}: {bets} bets | {strike:.1f}% | ROI {roi:+.1f}%")
    progress("")

conn.close()

progress("\n" + "=" * 100)
progress("CONCLUSION")
progress("=" * 100)
