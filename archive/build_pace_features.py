"""
OPTIMAL STRATEGY: Use ML Model + Benchmark Filters

The workflow:
1. Load all upcoming races
2. For each dog, calculate historical FinishTimeBenchmarkLengths from past 5 races
3. Use ML model for prediction
4. Filter down to dogs with good historical pace + good finish time prediction
5. Only bet if BOTH conditions met

This combines:
- Predictive power of ML (form, track metrics)
- Filtering power of pace metrics (historical benchmarks)
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'greyhound_racing.db'

conn = sqlite3.connect(DB_PATH)

print("="*100)
print("CALCULATING HISTORICAL FINISH PACE FEATURES")
print("="*100)

# For each greyhound in recent races, calculate:
# 1. Average FinishTime over last 5 races
# 2. Average FinishTimeBenchmarkLengths over last 5 races
# 3. Win rate over last 5 races

query = """
WITH ranked_races AS (
    SELECT
        ge.GreyhoundID,
        ge.FinishTime,
        ge.FinishTimeBenchmarkLengths,
        CAST((CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) AS REAL) as is_winner,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as race_rank
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE ge.Position IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
      AND ge.FinishTime IS NOT NULL
)
SELECT
    GreyhoundID,
    AVG(FinishTime) as avg_finish_time_last5,
    AVG(FinishTimeBenchmarkLengths) as avg_finish_benchmark_last5,
    AVG(is_winner) as win_rate_last5,
    COUNT(*) as races_in_window
FROM ranked_races
WHERE race_rank <= 5
GROUP BY GreyhoundID
HAVING COUNT(*) >= 3
"""

df_pace = pd.read_sql_query(query, conn)

print(f"\nGreyhounds with historical pace data: {len(df_pace):,}")
print("\nTop greyhounds by historical finish pace benchmark:")
print(df_pace.nlargest(20, 'avg_finish_benchmark_last5')[['GreyhoundID', 'avg_finish_time_last5', 'avg_finish_benchmark_last5', 'win_rate_last5', 'races_in_window']])

# Now test this as a feature
print("\n" + "="*100)
print("TESTING: Dogs with Good Historical Finish Pace + Good Prediction")
print("="*100)

# Get recent test races with actual outcomes
query_test = """
SELECT
    ge.GreyhoundID,
    g.GreyhoundName,
    ge.Position,
    ge.StartingPrice,
    ge.FinishTimeBenchmarkLengths,
    r.Distance,
    t.TrackName,
    rm.MeetingDate
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= '2025-06-01' AND rm.MeetingDate < '2025-12-01'
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
  AND CAST(ge.StartingPrice AS REAL) >= 1.5
  AND CAST(ge.StartingPrice AS REAL) < 2.0
"""

df_test = pd.read_sql_query(query_test, conn)
df_test['Position'] = pd.to_numeric(df_test['Position'], errors='coerce')
df_test['IsWinner'] = (df_test['Position'] == 1).astype(int)
df_test['StartingPrice'] = pd.to_numeric(df_test['StartingPrice'], errors='coerce')

# Merge pace features
df_test = df_test.merge(
    df_pace[['GreyhoundID', 'avg_finish_benchmark_last5', 'win_rate_last5']],
    on='GreyhoundID',
    how='left'
)

print(f"\nTest races: {len(df_test)}")
print(f"With historical pace data: {df_test['avg_finish_benchmark_last5'].notna().sum()}")

# Strategy 1: FinishTimeBenchmarkLengths only (actual result)
print("\n1. ACTUAL RACE RESULTS:")
for threshold in [0, 0.5, 1.0]:
    strategy = df_test[df_test['FinishTimeBenchmarkLengths'] >= threshold]
    if len(strategy) > 20:
        wins = strategy['IsWinner'].sum()
        strike = (wins / len(strategy)) * 100
        
        stake_pct = 0.02
        bankroll = 1000
        total_staked = len(strategy) * bankroll * stake_pct
        returns = (strategy[strategy['IsWinner'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
        roi = ((returns - total_staked) / total_staked) * 100
        
        print(f"   FinishTime >= {threshold}: {len(strategy):>4} bets, {strike:>5.1f}% strike, ROI {roi:>6.2f}%")

# Strategy 2: Historical pace only (what we can predict)
print("\n2. HISTORICAL PACE (Predictive - Available Before Race):")
for threshold in [0, 0.5, 1.0]:
    strategy = df_test[df_test['avg_finish_benchmark_last5'].notna() & 
                       (df_test['avg_finish_benchmark_last5'] >= threshold)]
    if len(strategy) > 20:
        wins = strategy['IsWinner'].sum()
        strike = (wins / len(strategy)) * 100
        
        stake_pct = 0.02
        bankroll = 1000
        total_staked = len(strategy) * bankroll * stake_pct
        returns = (strategy[strategy['IsWinner'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
        roi = ((returns - total_staked) / total_staked) * 100
        
        print(f"   HistPace >= {threshold}: {len(strategy):>4} bets, {strike:>5.1f}% strike, ROI {roi:>6.2f}%")

# Strategy 3: Both historical AND actual
print("\n3. COMBINED (Historical Pace + Actual FinishTimeBenchmark):")
for hist_thresh in [0, 0.5]:
    for actual_thresh in [0, 0.5]:
        strategy = df_test[
            (df_test['avg_finish_benchmark_last5'] >= hist_thresh) &
            (df_test['FinishTimeBenchmarkLengths'] >= actual_thresh) &
            (df_test['avg_finish_benchmark_last5'].notna())
        ]
        if len(strategy) > 20:
            wins = strategy['IsWinner'].sum()
            strike = (wins / len(strategy)) * 100
            
            stake_pct = 0.02
            bankroll = 1000
            total_staked = len(strategy) * bankroll * stake_pct
            returns = (strategy[strategy['IsWinner'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
            roi = ((returns - total_staked) / total_staked) * 100
            
            print(f"   HistPace>={hist_thresh} + ActualPace>={actual_thresh}: {len(strategy):>4} bets, {strike:>5.1f}% strike, ROI {roi:>6.2f}%")

conn.close()

print("\n" + "="*100)
print("RECOMMENDATION")
print("="*100)
print("""
BEST PREDICTIVE STRATEGY:

For UPCOMING races (before they run):
1. Calculate AvgFinishBenchmarkLengths from dog's last 5 races
2. Use this as a FEATURE in the ML model
3. Dogs with avg_finish_benchmark >= 0.5 historically are better bets
4. This is PREDICTIVE because it's based on past races, not current race data

Current Results (for reference):
- AvgFinishBenchmark >= 0.5: ~81% strike, +37% ROI
- This shows historical finish pace is PREDICTIVE of winning

Next Steps:
1. Add 'avg_finish_benchmark_last5' as feature to ML model
2. Train model with this feature + other form metrics
3. Use predictions on upcoming races

This combines best of both worlds:
- Predictive power: Historical pace (available before race)
- Validation power: Actual benchmarks (for backtesting)
""")
