"""
BACKTEST: Find Optimal Confidence Threshold
Tests multiple score thresholds to find the sweet spot
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import time

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

progress("="*100)
progress("BACKTEST: Testing Optimal Confidence Threshold")
progress("="*100)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Get all historical races with pace and form metrics
query = """
WITH dog_pace_history AS (
    SELECT 
        ge.GreyhoundID,
        g.GreyhoundName,
        rm.MeetingDate,
        (ge.FinishTimeBenchmarkLengths + rm.MeetingAvgBenchmarkLengths) as TotalFinishBench,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE ge.FinishTimeBenchmarkLengths IS NOT NULL
      AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
      AND ge.Position IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
),

dog_pace_avg AS (
    SELECT 
        GreyhoundID,
        GreyhoundName,
        AVG(CASE WHEN RaceNum <= 5 THEN TotalFinishBench END) as HistoricalPaceAvg,
        COUNT(DISTINCT CASE WHEN RaceNum <= 5 THEN MeetingDate END) as PacesUsed
    FROM dog_pace_history
    GROUP BY GreyhoundID
    HAVING PacesUsed >= 5
),

dog_recent_form AS (
    SELECT 
        ge.GreyhoundID,
        (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as IsWinner,
        ge.StartingPrice,
        rm.MeetingDate,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE ge.Position IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
      AND ge.StartingPrice IS NOT NULL
      AND rm.MeetingDate >= '2025-01-01'
),

dog_form_last_5 AS (
    SELECT 
        GreyhoundID,
        SUM(CASE WHEN IsWinner = 1 THEN 1 ELSE 0 END) as RecentWins,
        COUNT(*) as RecentRaces,
        ROUND(100.0 * SUM(CASE WHEN IsWinner = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as RecentWinRate
    FROM dog_recent_form
    WHERE RaceNum <= 5
    GROUP BY GreyhoundID
)

SELECT 
    dpa.GreyhoundName,
    dpa.HistoricalPaceAvg,
    COALESCE(dfl.RecentWins, 0) as RecentWins,
    COALESCE(dfl.RecentRaces, 0) as RecentRaces,
    COALESCE(dfl.RecentWinRate, 0) as RecentWinRate,
    drf.IsWinner,
    drf.StartingPrice,
    drf.MeetingDate,
    ROW_NUMBER() OVER (PARTITION BY dpa.GreyhoundID ORDER BY drf.MeetingDate DESC) as RaceNum
FROM dog_pace_avg dpa
JOIN dog_recent_form drf ON dpa.GreyhoundID = drf.GreyhoundID
LEFT JOIN dog_form_last_5 dfl ON dpa.GreyhoundID = dfl.GreyhoundID
WHERE drf.RaceNum >= 1
  AND drf.MeetingDate >= '2025-01-01'
"""

progress("\nLoading 2025 race data...", indent=1)
start = time.time()
df = pd.read_sql_query(query, conn)
elapsed = time.time() - start
progress(f"Loaded {len(df):,} races in {elapsed:.1f}s", indent=1)

conn.close()

# Clean data
df['IsWinner'] = pd.to_numeric(df['IsWinner'], errors='coerce').fillna(0)
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['HistoricalPaceAvg'] = pd.to_numeric(df['HistoricalPaceAvg'], errors='coerce')
df['RecentWinRate'] = pd.to_numeric(df['RecentWinRate'], errors='coerce')

df = df.dropna(subset=['IsWinner', 'HistoricalPaceAvg', 'StartingPrice'])

progress(f"After cleaning: {len(df):,} races", indent=1)

# ============================================================================
# TEST MULTIPLE THRESHOLDS
# ============================================================================
progress("\n" + "="*100)
progress("TESTING MULTIPLE CONFIDENCE THRESHOLDS")
progress("="*100)

# Calculate normalized scores
pace_min = df['HistoricalPaceAvg'].min()
pace_max = df['HistoricalPaceAvg'].max()
df['PaceScore'] = (df['HistoricalPaceAvg'] - pace_min) / (pace_max - pace_min)
df['FormScore'] = df['RecentWinRate'] / 100.0
df['WeightedScore'] = (df['PaceScore'] * 0.7) + (df['FormScore'] * 0.3)

# Test different thresholds
thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]
results = []

for threshold in thresholds:
    subset = df[
        (df['WeightedScore'] >= threshold) &
        (df['StartingPrice'] >= 1.50) &
        (df['StartingPrice'] <= 5.00)
    ].copy()
    
    if len(subset) == 0:
        continue
    
    wins = subset['IsWinner'].sum()
    bets = len(subset)
    strike_rate = wins / bets * 100
    avg_odds = subset['StartingPrice'].mean()
    total_return = (wins * avg_odds) + (bets - wins) * 0.0
    roi = ((total_return - bets) / bets) * 100
    
    results.append({
        'threshold': threshold,
        'bets': bets,
        'wins': wins,
        'strike': strike_rate,
        'avg_odds': avg_odds,
        'roi': roi
    })
    
    progress(f"Threshold {threshold:.2f}: {bets:,} bets, {strike_rate:.1f}% strike, {roi:+.1f}% ROI", indent=1)

# Display table
progress("\n" + "="*100)
progress("SUMMARY TABLE")
progress("="*100)

print("\n{:<15} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
    "Threshold", "Bets", "Wins", "Strike %", "Avg Odds", "ROI %"
))
print("-" * 75)

for r in results:
    print("{:<15} {:<12,} {:<12,} {:<12.1f} ${:<11.2f} {:<12.1f}".format(
        f"{r['threshold']:.2f}",
        r['bets'],
        r['wins'],
        r['strike'],
        r['avg_odds'],
        r['roi']
    ))

# Find best ROI
if results:
    best = max(results, key=lambda x: x['roi'])
    progress(f"\nOptimal Threshold: {best['threshold']:.2f}", indent=1)
    progress(f"  Best ROI: {best['roi']:+.1f}%", indent=2)
    progress(f"  Strike Rate: {best['strike']:.1f}%", indent=2)
    progress(f"  Bets: {best['bets']:,}", indent=2)

progress("\n" + "="*100)
