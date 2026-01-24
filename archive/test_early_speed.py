"""
Test early speed filtering using SplitBenchmarkLengths
Positive = Faster than benchmark (good early speed)
Negative = Slower than benchmark (bad early speed)
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'greyhound_racing.db'

conn = sqlite3.connect(DB_PATH)

print("="*80)
print("EARLY SPEED STRATEGY: SplitBenchmarkLengths")
print("="*80)

# Load data from test period with good odds range
query = """
SELECT
    t.TrackName, ge.Box, ge.Position, ge.StartingPrice, 
    ge.SplitBenchmarkLengths, ge.Weight, r.Distance
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= '2025-01-01' AND rm.MeetingDate < '2025-12-01'
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
"""

df = pd.read_sql_query(query, conn)
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['IsWinner'] = (df['Position'] == 1).astype(int)
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')

# Filter to $1.50-$2.00
df = df[(df['StartingPrice'] >= 1.5) & (df['StartingPrice'] < 2.0)]

print(f"\nTotal races at $1.50-$2.00: {len(df)}")
print(f"Overall win rate: {df['IsWinner'].mean()*100:.1f}%")

# Check SplitBenchmarkLengths availability
null_count = df['SplitBenchmarkLengths'].isna().sum()
print(f"SplitBenchmarkLengths data availability: {len(df) - null_count:,} / {len(df):,} ({(1 - null_count/len(df))*100:.1f}%)")

# Test 1: Dogs with POSITIVE SplitBenchmarkLengths (faster than benchmark)
print("\n" + "="*80)
print("TEST 1: POSITIVE SplitBenchmarkLengths (Faster than Benchmark)")
print("="*80)

df_clean = df[df['SplitBenchmarkLengths'].notna()].copy()

for threshold in [0, 0.5, 1.0, 1.5]:
    strategy = df_clean[df_clean['SplitBenchmarkLengths'] >= threshold]
    if len(strategy) > 20:
        wins = strategy['IsWinner'].sum()
        strike = (wins / len(strategy)) * 100
        
        # Calculate ROI
        stake_pct = 0.02
        bankroll = 1000
        total_staked = len(strategy) * bankroll * stake_pct
        returns = (strategy[strategy['IsWinner'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
        pnl = returns - total_staked
        roi = (pnl / total_staked) * 100 if total_staked > 0 else 0
        
        print(f"SplitBenchmarkLengths >= {threshold:>3}: {len(strategy):>5} bets, {strike:>5.1f}% strike, ROI {roi:>6.2f}%")

# Test 2: Dogs with NEGATIVE SplitBenchmarkLengths (slower than benchmark)
print("\n" + "="*80)
print("TEST 2: NEGATIVE SplitBenchmarkLengths (Slower than Benchmark)")
print("="*80)

for threshold in [0, -0.5, -1.0, -1.5]:
    strategy = df_clean[df_clean['SplitBenchmarkLengths'] <= threshold]
    if len(strategy) > 20:
        wins = strategy['IsWinner'].sum()
        strike = (wins / len(strategy)) * 100
        
        stake_pct = 0.02
        bankroll = 1000
        total_staked = len(strategy) * bankroll * stake_pct
        returns = (strategy[strategy['IsWinner'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
        pnl = returns - total_staked
        roi = (pnl / total_staked) * 100 if total_staked > 0 else 0
        
        print(f"SplitBenchmarkLengths <= {threshold:>3}: {len(strategy):>5} bets, {strike:>5.1f}% strike, ROI {roi:>6.2f}%")

# Test 3: Combine with box position (Box <= 2 + good early speed)
print("\n" + "="*80)
print("TEST 3: Box <= 2 AND SplitBenchmarkLengths >= 0 (Good Position + Good Early Speed)")
print("="*80)

for split_threshold in [0, 0.5, 1.0]:
    strategy = df_clean[(df_clean['Box'] <= 2) & (df_clean['SplitBenchmarkLengths'] >= split_threshold)]
    if len(strategy) > 20:
        wins = strategy['IsWinner'].sum()
        strike = (wins / len(strategy)) * 100
        
        stake_pct = 0.02
        bankroll = 1000
        total_staked = len(strategy) * bankroll * stake_pct
        returns = (strategy[strategy['IsWinner'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
        pnl = returns - total_staked
        roi = (pnl / total_staked) * 100 if total_staked > 0 else 0
        
        print(f"Box<=2 + Split>={split_threshold:>3}: {len(strategy):>5} bets, {strike:>5.1f}% strike, ROI {roi:>6.2f}%")

# Test 4: Distribution analysis
print("\n" + "="*80)
print("DISTRIBUTION ANALYSIS")
print("="*80)

print(f"\nSplitBenchmarkLengths statistics:")
print(f"  Min: {df_clean['SplitBenchmarkLengths'].min():.2f}")
print(f"  Max: {df_clean['SplitBenchmarkLengths'].max():.2f}")
print(f"  Mean: {df_clean['SplitBenchmarkLengths'].mean():.2f}")
print(f"  Median: {df_clean['SplitBenchmarkLengths'].median():.2f}")
print(f"  Std Dev: {df_clean['SplitBenchmarkLengths'].std():.2f}")

# Win rates by quartile
print(f"\nWin rate by SplitBenchmarkLengths quartile:")
for i in range(4):
    q_low = df_clean['SplitBenchmarkLengths'].quantile(i/4)
    q_high = df_clean['SplitBenchmarkLengths'].quantile((i+1)/4)
    subset = df_clean[(df_clean['SplitBenchmarkLengths'] >= q_low) & 
                      (df_clean['SplitBenchmarkLengths'] <= q_high)]
    win_rate = subset['IsWinner'].mean() * 100
    print(f"  Q{i+1} ({q_low:>6.2f} to {q_high:>6.2f}): {win_rate:>5.1f}% ({len(subset):>5} bets)")

conn.close()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("SplitBenchmarkLengths is a REAL early speed metric!")
print("Positive = faster than field = better")
print("Negative = slower than field = worse")
print("\nNext: Test if combining with Box position creates a viable strategy")
