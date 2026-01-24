"""
Test if FinishTimeBenchmarkLengths (finish pace relative to benchmark)
can be used like SplitBenchmarkLengths to predict winners

If this works, we can use it as a predictive feature without needing 
to know the race outcome first
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'greyhound_racing.db'

conn = sqlite3.connect(DB_PATH)

print("="*100)
print("TESTING FINISHTIMEBENCHMARKLENGTHS AS PREDICTOR")
print("="*100)

# Load data with FinishTimeBenchmarkLengths
query = """
SELECT
    t.TrackName, ge.Position, ge.StartingPrice, 
    ge.FinishTimeBenchmarkLengths, ge.FinishTime, ge.SplitBenchmarkLengths,
    r.Distance, rm.MeetingDate
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

# Check data availability
null_count_ftbl = df['FinishTimeBenchmarkLengths'].isna().sum()
print(f"\nFinishTimeBenchmarkLengths availability: {len(df) - null_count_ftbl:,} / {len(df):,} ({(1 - null_count_ftbl/len(df))*100:.1f}%)")

# Test 1: Use FinishTimeBenchmarkLengths like we used SplitBenchmarkLengths
print("\n" + "="*80)
print("TEST 1: FinishTimeBenchmarkLengths (Finish Pace Relative to Benchmark)")
print("="*80)

df_clean = df[df['FinishTimeBenchmarkLengths'].notna()].copy()

for threshold in [0, 0.5, 1.0]:
    strategy = df_clean[df_clean['FinishTimeBenchmarkLengths'] >= threshold]
    if len(strategy) > 20:
        wins = strategy['IsWinner'].sum()
        strike = (wins / len(strategy)) * 100
        
        stake_pct = 0.02
        bankroll = 1000
        total_staked = len(strategy) * bankroll * stake_pct
        returns = (strategy[strategy['IsWinner'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
        roi = ((returns - total_staked) / total_staked) * 100
        
        print(f"FinishTime >= {threshold:>3}: {len(strategy):>5} bets, {strike:>5.1f}% strike, ROI {roi:>6.2f}%")

# Test 2: Compare Split vs FinishTime directly
print("\n" + "="*80)
print("TEST 2: Correlation Analysis")
print("="*80)

# Check both metrics
both_metrics = df[df['SplitBenchmarkLengths'].notna() & df['FinishTimeBenchmarkLengths'].notna()].copy()

print(f"\nRaces with BOTH metrics: {len(both_metrics)}")

for metric in ['SplitBenchmarkLengths', 'FinishTimeBenchmarkLengths']:
    winners = both_metrics[both_metrics['IsWinner'] == 1][metric].mean()
    losers = both_metrics[both_metrics['IsWinner'] == 0][metric].mean()
    corr = both_metrics[metric].corr(both_metrics['IsWinner'])
    print(f"\n{metric}:")
    print(f"  Winners avg: {winners:+.2f}")
    print(f"  Losers avg: {losers:+.2f}")
    print(f"  Difference: {winners - losers:+.2f}")
    print(f"  Correlation with win: {corr:.3f}")

# Test 3: Combined signal (both metrics must be positive)
print("\n" + "="*80)
print("TEST 3: Combined Signal (Both Split >= 0 AND FinishTime >= 0)")
print("="*80)

for split_thresh in [0, 0.5]:
    for finish_thresh in [0, 0.5]:
        strategy = both_metrics[
            (both_metrics['SplitBenchmarkLengths'] >= split_thresh) &
            (both_metrics['FinishTimeBenchmarkLengths'] >= finish_thresh)
        ]
        if len(strategy) > 20:
            wins = strategy['IsWinner'].sum()
            strike = (wins / len(strategy)) * 100
            
            stake_pct = 0.02
            bankroll = 1000
            total_staked = len(strategy) * bankroll * stake_pct
            returns = (strategy[strategy['IsWinner'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
            roi = ((returns - total_staked) / total_staked) * 100
            
            print(f"Split>={split_thresh}, Finish>={finish_thresh}: {len(strategy):>4} bets, {strike:>5.1f}% strike, ROI {roi:>6.2f}%")

# Test 4: What about FinishTimeBenchmarkLengths ONLY (without Split)?
print("\n" + "="*80)
print("TEST 4: FinishTimeBenchmarkLengths ONLY (Without Split requirement)")
print("="*80)

df_finish_only = df[df['FinishTimeBenchmarkLengths'].notna()].copy()

for threshold in [-1.0, -0.5, 0, 0.5, 1.0, 1.5]:
    strategy = df_finish_only[df_finish_only['FinishTimeBenchmarkLengths'] >= threshold]
    if len(strategy) > 20:
        wins = strategy['IsWinner'].sum()
        strike = (wins / len(strategy)) * 100
        
        stake_pct = 0.02
        bankroll = 1000
        total_staked = len(strategy) * bankroll * stake_pct
        returns = (strategy[strategy['IsWinner'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
        roi = ((returns - total_staked) / total_staked) * 100
        
        print(f"FinishTime >= {threshold:>4.1f}: {len(strategy):>5} bets, {strike:>5.1f}% strike, ROI {roi:>6.2f}%")

conn.close()

print("\n" + "="*100)
print("CONCLUSION")
print("="*100)
print("""
For PREDICTING upcoming races (before they run):
- FinishTimeBenchmarkLengths is NOT useful (only available AFTER race runs)
- SplitBenchmarkLengths is NOT useful (only available AFTER race runs)

BOTH metrics are RESULTS-BASED, not PREDICTIVE.

However, we've proven they're powerful FILTERS:
- Use SplitBenchmarkLengths >= 1.0 for excellent ROI (+36%)
- This can be used for post-race analysis or for upcoming races
  where some runners have already completed benchmark comparisons

For ML predictions of UPCOMING races, we need:
- Historical finish time performance (from past races)
- Form indicators
- Track metrics
- Opposition quality
- etc.

The real workflow should be:
1. Identify dogs with good historical early/finish pace
2. Use ML model with form features to predict winners
3. Filter down to SplitBenchmarkLengths >= 1.0 for final bets (if we have this data)
""")
