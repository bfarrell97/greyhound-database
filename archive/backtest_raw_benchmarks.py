"""
Backtest using RAW greyhound benchmarks only (no meeting adjustment)
- G OT ADJ = FinishTimeBenchmarkLengths (higher = faster than track benchmark)
- G First Sec ADJ = SplitBenchmarkLengths (higher = faster split than benchmark)

Last 5 races, weekly rolling window
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
progress("BACKTEST: Raw G OT ADJ & G First Sec ADJ (Last 5 Races)")
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
    """Get raw G OT ADJ and G First Sec ADJ averages"""
    query = f"""
    WITH ranked_races AS (
        SELECT 
            ge.GreyhoundID,
            rm.MeetingDate,
            t.TrackName,
            ge.FinishTimeBenchmarkLengths as G_OT_ADJ,
            ge.SplitBenchmarkLengths as G_Split_ADJ,
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
        G_OT_ADJ,
        G_Split_ADJ,
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
        
        # Raw G OT ADJ (weighted by track tier)
        g_ot = np.average(group['G_OT_ADJ'].values, weights=weights)
        
        # Raw G Split ADJ
        g_split = np.average(group['G_Split_ADJ'].fillna(0).values, weights=weights)
        
        # Recent win rate
        win_rate = group['Winner'].sum() / len(group) * 100
        
        results.append({
            'GreyhoundID': gid,
            'G_OT_ADJ_Avg': g_ot,
            'G_Split_ADJ_Avg': g_split,
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

# Rank by each metric within each race (higher = better/faster)
test_df['Rank_OT'] = test_df.groupby('RaceID')['G_OT_ADJ_Avg'].rank(ascending=False, method='first')
test_df['Rank_Split'] = test_df.groupby('RaceID')['G_Split_ADJ_Avg'].rank(ascending=False, method='first')

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
progress("ANALYSIS 1: G OT ADJ Rankings (Overall Time)")
progress("=" * 100)

for rank in [1, 2, 3]:
    subset = test_df[test_df['Rank_OT'] == rank]
    bets, strike, roi = calc_roi(subset)
    avg_odds = subset['StartingPrice'].mean()
    progress(f"Rank {rank} G_OT_ADJ: {bets:,} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 2: G First Sec ADJ Rankings (Split Time)")
progress("=" * 100)

for rank in [1, 2, 3]:
    subset = test_df[test_df['Rank_Split'] == rank]
    bets, strike, roi = calc_roi(subset)
    avg_odds = subset['StartingPrice'].mean()
    progress(f"Rank {rank} G_Split_ADJ: {bets:,} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 3: Combining OT and Split")
progress("=" * 100)

# Best at both
subset = test_df[(test_df['Rank_OT'] == 1) & (test_df['Rank_Split'] == 1)]
bets, strike, roi = calc_roi(subset)
avg_odds = subset['StartingPrice'].mean()
progress(f"Best OT + Best Split: {bets:,} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

# Top-2 at both
subset = test_df[(test_df['Rank_OT'] <= 2) & (test_df['Rank_Split'] <= 2)]
bets, strike, roi = calc_roi(subset)
avg_odds = subset['StartingPrice'].mean()
progress(f"Top-2 OT + Top-2 Split: {bets:,} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 4: With Favourite Filter")
progress("=" * 100)

subset = test_df[(test_df['Rank_OT'] == 1) & (test_df['IsFavourite'])]
bets, strike, roi = calc_roi(subset)
avg_odds = subset['StartingPrice'].mean()
progress(f"Best G_OT_ADJ + Favourite: {bets} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

subset = test_df[(test_df['Rank_OT'] == 1) & (~test_df['IsFavourite'])]
bets, strike, roi = calc_roi(subset)
avg_odds = subset['StartingPrice'].mean()
progress(f"Best G_OT_ADJ NOT Favourite: {bets} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

subset = test_df[(test_df['Rank_Split'] == 1) & (test_df['IsFavourite'])]
bets, strike, roi = calc_roi(subset)
avg_odds = subset['StartingPrice'].mean()
progress(f"Best G_Split_ADJ + Favourite: {bets} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

subset = test_df[(test_df['Rank_Split'] == 1) & (~test_df['IsFavourite'])]
bets, strike, roi = calc_roi(subset)
avg_odds = subset['StartingPrice'].mean()
progress(f"Best G_Split_ADJ NOT Favourite: {bets} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 5: Price Range Analysis")
progress("=" * 100)

for low, high in [(1.5, 2.5), (2.0, 3.0), (2.5, 4.0), (3.0, 5.0), (4.0, 7.0)]:
    subset = test_df[(test_df['Rank_OT'] == 1) & 
                     (test_df['StartingPrice'] >= low) & 
                     (test_df['StartingPrice'] < high)]
    bets, strike, roi = calc_roi(subset)
    progress(f"Best G_OT ${low:.1f}-${high:.1f}: {bets} bets | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "-" * 50)

for low, high in [(1.5, 2.5), (2.0, 3.0), (2.5, 4.0), (3.0, 5.0), (4.0, 7.0)]:
    subset = test_df[(test_df['Rank_Split'] == 1) & 
                     (test_df['StartingPrice'] >= low) & 
                     (test_df['StartingPrice'] < high)]
    bets, strike, roi = calc_roi(subset)
    progress(f"Best G_Split ${low:.1f}-${high:.1f}: {bets} bets | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 6: Box Number Combinations")
progress("=" * 100)

for box in [1, 2, 5, 8]:
    subset = test_df[(test_df['Rank_OT'] == 1) & (test_df['Box'] == box)]
    bets, strike, roi = calc_roi(subset)
    avg_odds = subset['StartingPrice'].mean()
    progress(f"Best G_OT + Box {box}: {bets} bets | ${avg_odds:.2f} avg | {strike:.1f}% | ROI {roi:+.1f}%")

progress("\n" + "=" * 100)
progress("ANALYSIS 7: Combined Best Filters")
progress("=" * 100)

results = []

configs = [
    ("Best OT + Fav", (test_df['Rank_OT'] == 1) & (test_df['IsFavourite'])),
    ("Best Split + Fav", (test_df['Rank_Split'] == 1) & (test_df['IsFavourite'])),
    ("Best OT + Best Split + Fav", (test_df['Rank_OT'] == 1) & (test_df['Rank_Split'] == 1) & (test_df['IsFavourite'])),
    ("Best OT + Box 1", (test_df['Rank_OT'] == 1) & (test_df['Box'] == 1)),
    ("Best Split + Box 1", (test_df['Rank_Split'] == 1) & (test_df['Box'] == 1)),
    ("Best OT + Best Split", (test_df['Rank_OT'] == 1) & (test_df['Rank_Split'] == 1)),
    ("Best OT + NOT Fav", (test_df['Rank_OT'] == 1) & (~test_df['IsFavourite'])),
    ("Best Split + NOT Fav", (test_df['Rank_Split'] == 1) & (~test_df['IsFavourite'])),
    ("Best OT + $1.5-2.5", (test_df['Rank_OT'] == 1) & (test_df['StartingPrice'] >= 1.5) & (test_df['StartingPrice'] < 2.5)),
    ("Best OT + $2-3", (test_df['Rank_OT'] == 1) & (test_df['StartingPrice'] >= 2) & (test_df['StartingPrice'] < 3)),
    ("Best OT + $2.5-4", (test_df['Rank_OT'] == 1) & (test_df['StartingPrice'] >= 2.5) & (test_df['StartingPrice'] < 4)),
    ("Best Split + $1.5-2.5", (test_df['Rank_Split'] == 1) & (test_df['StartingPrice'] >= 1.5) & (test_df['StartingPrice'] < 2.5)),
    ("Best Split + $2-3", (test_df['Rank_Split'] == 1) & (test_df['StartingPrice'] >= 2) & (test_df['StartingPrice'] < 3)),
    ("Best Split + $2.5-4", (test_df['Rank_Split'] == 1) & (test_df['StartingPrice'] >= 2.5) & (test_df['StartingPrice'] < 4)),
    ("Top-2 OT + Top-2 Split + Fav", (test_df['Rank_OT'] <= 2) & (test_df['Rank_Split'] <= 2) & (test_df['IsFavourite'])),
    ("Best OT + Fav + Box 1/2/8", (test_df['Rank_OT'] == 1) & (test_df['IsFavourite']) & (test_df['Box'].isin([1, 2, 8]))),
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
    progress(f"{name:40s} | {bets:5d} ({daily:5.1f}/day) | {strike:5.1f}% | ROI {roi:+6.1f}%")

conn.close()

progress("\n" + "=" * 100)
progress("SUMMARY")
progress("=" * 100)
progress("""
Using raw G OT ADJ and G First Sec ADJ (no meeting adjustment):
- G OT ADJ = FinishTimeBenchmarkLengths (greyhound's time vs track/distance benchmark)
- G Split ADJ = SplitBenchmarkLengths (greyhound's split vs track/distance benchmark)
- Higher values = faster dog
""")
