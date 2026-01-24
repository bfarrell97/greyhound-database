"""
BACKTEST: New Weighted Model (70% Pace + 30% Form)
Validates the new strategy against 2025 historical data
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
progress("BACKTEST: New Weighted Model (70% Pace + 30% Form)")
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

# Remove NZ tracks
excluded_tracks = ['Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North',
                  'Launceston', 'Hobart', 'Devonport']

df = df.dropna(subset=['IsWinner', 'HistoricalPaceAvg', 'StartingPrice'])

progress(f"After cleaning: {len(df):,} races", indent=1)

# ============================================================================
# TEST THE WEIGHTED MODEL
# ============================================================================
progress("\n" + "="*100)
progress("TESTING WEIGHTED MODEL: 70% Pace + 30% Form at threshold 0.6 (BROAD $1.50-$5.00 RANGE)")
progress("="*100)

# Calculate normalized scores
pace_min = df['HistoricalPaceAvg'].min()
pace_max = df['HistoricalPaceAvg'].max()
df['PaceScore'] = (df['HistoricalPaceAvg'] - pace_min) / (pace_max - pace_min)
df['FormScore'] = df['RecentWinRate'] / 100.0

# Weighted score
df['WeightedScore'] = (df['PaceScore'] * 0.7) + (df['FormScore'] * 0.3)

# Test at threshold 0.6 on $1.50-$5.00 odds (BROAD RANGE APPROVED)
subset = df[
    (df['WeightedScore'] >= 0.6) &
    (df['StartingPrice'] >= 1.50) &
    (df['StartingPrice'] <= 5.00)
].copy()

progress(f"\nDogs selected: {len(subset):,}", indent=1)
progress(f"Win rate: {(subset['IsWinner'].sum() / len(subset) * 100):.1f}%", indent=1)

wins = subset['IsWinner'].sum()
bets = len(subset)
strike_rate = wins / bets * 100
avg_odds = subset['StartingPrice'].mean()

# Calculate ROI
total_return = (wins * avg_odds) + (bets - wins) * 0.0
roi = ((total_return - bets) / bets) * 100

progress("\n" + "="*100)
progress("RESULTS")
progress("="*100)
progress(f"Total Bets: {bets:,}", indent=1)
progress(f"Total Wins: {wins:,}", indent=1)
progress(f"Strike Rate: {strike_rate:.1f}%", indent=1)
progress(f"Average Odds: ${avg_odds:.2f}", indent=1)
progress(f"Total Stake: ${bets:,.0f}", indent=1)
progress(f"Total Return: ${total_return:,.0f}", indent=1)
progress(f"Profit: ${total_return - bets:,.0f}", indent=1)
progress(f"ROI: {roi:+.1f}%", indent=1)

progress("\n" + "="*100)
progress("COMPARISON TO ORIGINAL +13% ROI CLAIM")
progress("="*100)

# Now test the original model (pace only, threshold 0.5) - same odds range for fair comparison
subset_original = df[
    (df['PaceScore'] >= 0.5) &
    (df['StartingPrice'] >= 1.50) &
    (df['StartingPrice'] <= 5.00)
].copy()

wins_orig = subset_original['IsWinner'].sum()
bets_orig = len(subset_original)
strike_rate_orig = wins_orig / bets_orig * 100
avg_odds_orig = subset_original['StartingPrice'].mean()
total_return_orig = (wins_orig * avg_odds_orig) + (bets_orig - wins_orig) * 0.0
roi_orig = ((total_return_orig - bets_orig) / bets_orig) * 100

progress(f"Original Model (Pace Only):", indent=1)
progress(f"  Bets: {bets_orig:,} | Wins: {wins_orig:,} ({strike_rate_orig:.1f}%) | ROI: {roi_orig:+.1f}%", indent=2)

progress(f"\nNew Model (70% Pace + 30% Form):", indent=1)
progress(f"  Bets: {bets:,} | Wins: {wins:,} ({strike_rate:.1f}%) | ROI: {roi:+.1f}%", indent=2)

improvement = roi - roi_orig
progress(f"\nImprovement: {improvement:+.1f}% ROI", indent=1)

if roi > roi_orig:
    progress("VERDICT: NEW MODEL IS BETTER", indent=1)
else:
    progress("VERDICT: Original model performs better", indent=1)

progress("\n" + "="*100)
