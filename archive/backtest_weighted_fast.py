"""
BACKTEST: New Weighted Model (70% Pace + 30% Form) - FAST WITH PROGRESS
Tests on historical data with detailed progress output
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import time
import sys

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

progress("="*100)
progress("BACKTEST: New Weighted Model (70% Pace + 30% Form) - WITH PROGRESS")
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

# Clean data with progress
progress("\nCleaning data...", indent=1)
df['IsWinner'] = pd.to_numeric(df['IsWinner'], errors='coerce').fillna(0)
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['HistoricalPaceAvg'] = pd.to_numeric(df['HistoricalPaceAvg'], errors='coerce')
df['RecentWinRate'] = pd.to_numeric(df['RecentWinRate'], errors='coerce')
df = df.dropna(subset=['IsWinner', 'HistoricalPaceAvg', 'StartingPrice'])
progress(f"After cleaning: {len(df):,} races", indent=1)

# Calculate scores with progress
progress("\nCalculating weighted scores...", indent=1)
pace_min = df['HistoricalPaceAvg'].min()
pace_max = df['HistoricalPaceAvg'].max()

if pace_max - pace_min == 0:
    df['PaceScore'] = 0.5
else:
    df['PaceScore'] = (df['HistoricalPaceAvg'] - pace_min) / (pace_max - pace_min)

df['FormScore'] = df['RecentWinRate'] / 100.0
df['WeightedScore'] = (df['PaceScore'] * 0.7) + (df['FormScore'] * 0.3)
progress(f"Score calculation complete", indent=1)

# ============================================================================
# TEST THE WEIGHTED MODEL
# ============================================================================
progress("\n" + "="*100)
progress("TESTING WEIGHTED MODEL: 70% Pace + 30% Form at threshold 0.6")
progress("="*100)

progress("\nFiltering to score >= 0.6 and odds $1.50-$5.00...", indent=1)
subset = df[
    (df['WeightedScore'] >= 0.6) &
    (df['StartingPrice'] >= 1.50) &
    (df['StartingPrice'] <= 5.00)
].copy()

progress(f"Dogs selected: {len(subset):,} ({100*len(subset)/len(df):.1f}% of all races)", indent=1)

wins = subset['IsWinner'].sum()
bets = len(subset)
strike_rate = wins / bets * 100 if bets > 0 else 0
avg_odds = subset['StartingPrice'].mean()

# Calculate ROI
total_return = (wins * avg_odds) + (bets - wins) * 0.0
roi = ((total_return - bets) / bets) * 100 if bets > 0 else 0

progress(f"Win rate: {strike_rate:.1f}%", indent=1)

progress("\n" + "="*100)
progress("RESULTS - ALL QUALIFIED (Score >= 0.6)")
progress("="*100)
progress(f"Total Bets: {bets:,}", indent=1)
progress(f"Total Wins: {wins:,}", indent=1)
progress(f"Strike Rate: {strike_rate:.1f}%", indent=1)
progress(f"Average Odds: ${avg_odds:.2f}", indent=1)
progress(f"Total Stake: ${bets:,.0f}", indent=1)
progress(f"Total Return: ${total_return:,.0f}", indent=1)
progress(f"Profit: ${total_return - bets:,.0f}", indent=1)
progress(f"ROI: {roi:+.1f}%", indent=1)

# ============================================================================
# TEST HIGH CONFIDENCE ONLY
# ============================================================================
progress("\n" + "="*100)
progress("TESTING HIGH CONFIDENCE ONLY: Score >= 0.75")
progress("="*100)

progress("\nFiltering to score >= 0.75 and odds $1.50-$5.00...", indent=1)
subset_hc = df[
    (df['WeightedScore'] >= 0.75) &
    (df['StartingPrice'] >= 1.50) &
    (df['StartingPrice'] <= 5.00)
].copy()

progress(f"Dogs selected: {len(subset_hc):,} ({100*len(subset_hc)/len(df):.1f}% of all races)", indent=1)

if len(subset_hc) > 0:
    wins_hc = subset_hc['IsWinner'].sum()
    bets_hc = len(subset_hc)
    strike_rate_hc = wins_hc / bets_hc * 100 if bets_hc > 0 else 0
    avg_odds_hc = subset_hc['StartingPrice'].mean()
    total_return_hc = (wins_hc * avg_odds_hc) + (bets_hc - wins_hc) * 0.0
    roi_hc = ((total_return_hc - bets_hc) / bets_hc) * 100 if bets_hc > 0 else 0
    
    progress(f"Win rate: {strike_rate_hc:.1f}%", indent=1)
    
    progress("\n" + "="*100)
    progress("RESULTS - HIGH CONFIDENCE ONLY (Score >= 0.75)")
    progress("="*100)
    progress(f"Total Bets: {bets_hc:,}", indent=1)
    progress(f"Total Wins: {wins_hc:,}", indent=1)
    progress(f"Strike Rate: {strike_rate_hc:.1f}%", indent=1)
    progress(f"Average Odds: ${avg_odds_hc:.2f}", indent=1)
    progress(f"Total Stake: ${bets_hc:,.0f}", indent=1)
    progress(f"Total Return: ${total_return_hc:,.0f}", indent=1)
    progress(f"Profit: ${total_return_hc - bets_hc:,.0f}", indent=1)
    progress(f"ROI: {roi_hc:+.1f}%", indent=1)
else:
    progress("No dogs met high confidence criteria!", indent=1)
    roi_hc = 0

# ============================================================================
# COMPARISON
# ============================================================================
progress("\n" + "="*100)
progress("COMPARISON")
progress("="*100)
progress(f"All Qualified (0.60+): {bets:,} bets, {strike_rate:.1f}% strike, {roi:+.1f}% ROI", indent=1)
if len(subset_hc) > 0:
    progress(f"High Confidence (0.75+): {bets_hc:,} bets, {strike_rate_hc:.1f}% strike, {roi_hc:+.1f}% ROI", indent=1)
    diff = roi - roi_hc
    progress(f"\nDifference: {diff:+.1f}pp ROI", indent=1)
    if roi > roi_hc:
        progress("VERDICT: All Qualified is BETTER", indent=1)
    else:
        progress("VERDICT: High Confidence is BETTER", indent=1)

progress("\n" + "="*100)
