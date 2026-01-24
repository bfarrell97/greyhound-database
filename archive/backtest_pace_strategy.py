"""
BACKTEST: Pace-Based Betting Strategy with Progress Updates
Tests the ROI claim of 65% strike rate, +13% ROI
Provides detailed progress updates so you can monitor execution
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import time

def progress(msg, indent=0):
    """Print timestamped progress message"""
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

progress("="*100)
progress("PACE-BASED BETTING STRATEGY BACKTEST")
progress("="*100)
progress(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

DB_PATH = 'greyhound_racing.db'

progress("\nSTEP 1: Connecting to database...")
conn = sqlite3.connect(DB_PATH)
progress("Connected successfully", indent=1)

progress("\nSTEP 2: Loading 2025 race data...")
query = """
SELECT
    ge.EntryID, ge.GreyhoundID, g.GreyhoundName, t.TrackName,
    rm.MeetingDate, r.RaceNumber, r.Distance, ge.Box, ge.Weight,
    ge.Position, ge.StartingPrice, ge.FinishTimeBenchmarkLengths,
    rm.MeetingAvgBenchmarkLengths
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= '2025-01-01' AND rm.MeetingDate <= '2025-12-09'
    AND ge.Position IS NOT NULL
    AND ge.Position NOT IN ('DNF', 'SCR')
    AND ge.StartingPrice IS NOT NULL
    AND ge.FinishTimeBenchmarkLengths IS NOT NULL
    AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
ORDER BY rm.MeetingDate, t.TrackName, r.RaceNumber, ge.Box
"""

start_time = time.time()
df = pd.read_sql_query(query, conn)
elapsed = time.time() - start_time
progress(f"Loaded {len(df):,} entries in {elapsed:.1f}s", indent=1)

progress("\nSTEP 3: Filtering to Australian tracks...")
excluded_tracks = ['Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North',
                  'Launceston', 'Hobart', 'Devonport']
before_filter = len(df)
df = df[~df['TrackName'].isin(excluded_tracks)]
df = df[~df['TrackName'].str.contains('NZ', na=False, case=False)]
df = df[~df['TrackName'].str.contains('TAS', na=False, case=False)]
progress(f"Filtered {before_filter - len(df):,} non-Australian entries, {len(df):,} remaining", indent=1)

progress("\nSTEP 4: Processing race results...")
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['IsWinner'] = (df['Position'] == 1).astype(int)
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
winners = df['IsWinner'].sum()
baseline = df['IsWinner'].mean() * 100
progress(f"Winners: {winners:,} (baseline strike: {baseline:.2f}%)", indent=1)

progress("\nSTEP 5: Loading historical pace data...")
historical_query = """
SELECT ge.GreyhoundID, ge.FinishTimeBenchmarkLengths,
       rm.MeetingAvgBenchmarkLengths, rm.MeetingDate
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE ge.Position IS NOT NULL AND ge.Position NOT IN ('DNF', 'SCR')
  AND ge.FinishTimeBenchmarkLengths IS NOT NULL
  AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
ORDER BY ge.GreyhoundID, rm.MeetingDate DESC
"""

start_time = time.time()
hist_df = pd.read_sql_query(historical_query, conn)
elapsed = time.time() - start_time
progress(f"Loaded {len(hist_df):,} historical entries in {elapsed:.1f}s", indent=1)

progress("\nSTEP 6: Calculating historical pace...")

def calculate_historical_pace(row, hist_df):
    dog_id = row['GreyhoundID']
    race_dt = pd.to_datetime(row['MeetingDate'])
    dog_hist = hist_df[hist_df['GreyhoundID'] == dog_id].copy()
    dog_hist['MeetingDate'] = pd.to_datetime(dog_hist['MeetingDate'])
    before_race = dog_hist[dog_hist['MeetingDate'] < race_dt]
    if len(before_race) == 0:
        return np.nan
    last_5 = before_race.head(5)
    pace_values = last_5['FinishTimeBenchmarkLengths'] + last_5['MeetingAvgBenchmarkLengths']
    if len(pace_values) > 0:
        return pace_values.mean()
    return np.nan

start_time = time.time()
paces = []
for idx, (_, row) in enumerate(df.iterrows()):
    if (idx + 1) % 10000 == 0:
        elapsed = time.time() - start_time
        pct = ((idx + 1) / len(df)) * 100
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        eta = (len(df) - idx - 1) / rate if rate > 0 else 0
        progress(f"Progress: {idx+1:,}/{len(df):,} ({pct:.1f}%) ETA {eta:.0f}s", indent=1)
    pace = calculate_historical_pace(row, hist_df)
    paces.append(pace)

df['HistoricalPace'] = paces
elapsed = time.time() - start_time
progress(f"Pace calculated in {elapsed:.1f}s", indent=1)

progress("\nSTEP 7: Filtering to dogs with historical pace...")
df_with_pace = df[df['HistoricalPace'].notna()].copy()
progress(f"Dogs with pace data: {len(df_with_pace):,}", indent=1)

progress("\nSTEP 8: Filtering to optimal odds ($1.50-$2.00)...")
df_filtered = df_with_pace[
    (df_with_pace['StartingPrice'] >= 1.50) & 
    (df_with_pace['StartingPrice'] <= 2.00)
].copy()
baseline_range = df_filtered['IsWinner'].mean() * 100
progress(f"Dogs in range: {len(df_filtered):,} (baseline strike: {baseline_range:.2f}%)", indent=1)

progress("\nSTEP 9: Testing pace thresholds...")
thresholds = [0.0, 0.25, 0.5, 1.0, 1.5]
results = []

for threshold in thresholds:
    subset = df_filtered[df_filtered['HistoricalPace'] >= threshold].copy()
    if len(subset) == 0:
        continue
    
    wins = subset['IsWinner'].sum()
    bets = len(subset)
    strike = (wins / bets * 100) if bets > 0 else 0
    total_stake = bets * 1.0
    total_return = (subset['StartingPrice'] * subset['IsWinner']).sum()
    roi = ((total_return - total_stake) / total_stake * 100) if total_stake > 0 else 0
    profit = total_return - total_stake
    
    results.append({
        'Threshold': threshold,
        'Bets': bets,
        'Wins': wins,
        'Strike%': strike,
        'Profit': profit,
        'ROI%': roi
    })
    
    progress(f"Pace >= {threshold:.2f}: {bets:,} bets, {wins:,} wins, {strike:.2f}% strike, ROI {roi:+.2f}%", indent=1)

conn.close()

progress("\n" + "="*100)
progress("RESULTS SUMMARY")
progress("="*100)

results_df = pd.DataFrame(results)
print("\nThreshold | Bets  | Wins  | Strike% | Profit   | ROI")
print("-" * 60)
for _, row in results_df.iterrows():
    print(f"{row['Threshold']:9.2f} | {int(row['Bets']):5d} | {int(row['Wins']):5d} | {row['Strike%']:7.2f} | ${row['Profit']:8.2f} | {row['ROI%']:6.2f}%")

progress("\nDETAILED ANALYSIS: Pace >= 0.5 (Recommended)")
progress("="*100)

recommended = df_filtered[df_filtered['HistoricalPace'] >= 0.5].copy()
if len(recommended) > 0:
    wins = recommended['IsWinner'].sum()
    bets = len(recommended)
    strike = (wins / bets * 100)
    total_stake = bets * 1.0
    total_return = (recommended['StartingPrice'] * recommended['IsWinner']).sum()
    roi = ((total_return - total_stake) / total_stake * 100)
    profit = total_return - total_stake
    
    progress(f"\nTotal bets: {bets:,} | Wins: {wins:,} | Strike: {strike:.2f}%")
    progress(f"Total stake: ${total_stake:,.2f}")
    progress(f"Total return: ${total_return:,.2f}")
    progress(f"Net profit: ${profit:+,.2f}")
    progress(f"ROI: {roi:+.2f}% (Expected: +13.00%, Diff: {roi-13:+.2f}%)")
    
    bets_per_day = bets / 340
    profit_per_day = profit / 340
    progress(f"\nDaily average: {bets_per_day:.1f} bets, ${profit_per_day:+.2f} profit")
    
    if roi < 0:
        progress("\n*** CRITICAL *** NEGATIVE ROI DETECTED!")
        progress("This contradicts the +13% ROI claim from prior analysis.")
        progress("Investigating possible causes...")
    else:
        progress(f"\nResult: ROI is {roi:+.2f}% vs +13% target")

progress(f"\nBacktest completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
progress("="*100)
