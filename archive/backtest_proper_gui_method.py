"""
PROPER BACKTEST: Using exact same method as GUI
Tests the weighted model on upcoming races (UpcomingBettingRaces/UpcomingBettingRunners)
instead of historical races (GreyhoundEntries)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

progress("="*100)
progress("PROPER BACKTEST: GUI Method (UpcomingBettingRaces/UpcomingBettingRunners)")
progress("="*100)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Query using exact same method as GUI
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
        GreyhoundName,
        AVG(CASE WHEN RaceNum <= 5 THEN TotalFinishBench END) as HistoricalPaceAvg,
        COUNT(DISTINCT CASE WHEN RaceNum <= 5 THEN MeetingDate END) as PacesUsed
    FROM dog_pace_history
    GROUP BY GreyhoundName
    HAVING PacesUsed >= 5
),

dog_recent_form AS (
    SELECT 
        g.GreyhoundName,
        (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as IsWinner,
        rm.MeetingDate,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE ge.Position IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
),

dog_form_last_5 AS (
    SELECT 
        GreyhoundName,
        SUM(CASE WHEN IsWinner = 1 THEN 1 ELSE 0 END) as RecentWins,
        COUNT(*) as RecentRaces,
        ROUND(100.0 * SUM(CASE WHEN IsWinner = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as RecentWinRate
    FROM dog_recent_form
    WHERE RaceNum <= 5
    GROUP BY GreyhoundName
)

SELECT 
    ubr.GreyhoundName,
    ubr.BoxNumber as Box,
    ubr.CurrentOdds as Odds,
    COALESCE(dfl.RecentWins, 0) as RecentWins,
    COALESCE(dfl.RecentRaces, 0) as RecentRaces,
    COALESCE(dfl.RecentWinRate, 0) as RecentWinRate,
    COALESCE(dpa.HistoricalPaceAvg, 0) as HistoricalPaceAvg,
    ubr2.TrackName,
    ubr2.RaceNumber,
    DATE(ubr2.MeetingDate) as MeetingDate,
    ubr2.Distance
FROM UpcomingBettingRunners ubr
JOIN UpcomingBettingRaces ubr2 ON ubr.UpcomingBettingRaceID = ubr2.UpcomingBettingRaceID
LEFT JOIN dog_pace_avg dpa ON ubr.GreyhoundName = dpa.GreyhoundName
LEFT JOIN dog_form_last_5 dfl ON ubr.GreyhoundName = dfl.GreyhoundName
WHERE ubr.CurrentOdds IS NOT NULL
  AND ubr2.MeetingDate IS NOT NULL
ORDER BY ubr2.MeetingDate, ubr2.RaceNumber, ubr.BoxNumber
"""

progress("\nLoading upcoming betting races...", indent=1)
start = time.time()
df = pd.read_sql_query(query, conn)
elapsed = time.time() - start
progress(f"Loaded {len(df):,} runners in {elapsed:.1f}s", indent=1)

if len(df) == 0:
    progress("No data in UpcomingBettingRaces/UpcomingBettingRunners!", indent=1)
    progress("This table may not have historical data for backtesting.", indent=1)
    conn.close()
    exit(1)

# Check date range
progress(f"\nDate range: {df['MeetingDate'].min()} to {df['MeetingDate'].max()}", indent=1)
progress(f"Unique dates: {df['MeetingDate'].nunique():,}", indent=1)

# Clean data types
df['Odds'] = pd.to_numeric(df['Odds'], errors='coerce')
df['HistoricalPaceAvg'] = pd.to_numeric(df['HistoricalPaceAvg'], errors='coerce')
df['RecentWinRate'] = pd.to_numeric(df['RecentWinRate'], errors='coerce')
df['Box'] = pd.to_numeric(df['Box'], errors='coerce')

# Remove NaN rows
df = df.dropna(subset=['Odds', 'HistoricalPaceAvg'])

progress(f"After cleaning: {len(df):,} runners", indent=1)

# Calculate weighted scores (exact same as GUI)
pace_min = df['HistoricalPaceAvg'].min()
pace_max = df['HistoricalPaceAvg'].max()

if pace_max - pace_min == 0:
    df['PaceScore'] = 0.5
else:
    df['PaceScore'] = (df['HistoricalPaceAvg'] - pace_min) / (pace_max - pace_min)

df['FormScore'] = df['RecentWinRate'] / 100.0
df['WeightedScore'] = (df['PaceScore'] * 0.7) + (df['FormScore'] * 0.3)

# ============================================================================
# TEST THE WEIGHTED MODEL
# ============================================================================
progress("\n" + "="*100)
progress("TESTING WEIGHTED MODEL: 70% Pace + 30% Form at threshold 0.6 ($1.50-$5.00)")
progress("="*100)

# Filter to score >= 0.6 and odds $1.50-$5.00
subset = df[
    (df['WeightedScore'] >= 0.6) &
    (df['Odds'] >= 1.50) &
    (df['Odds'] <= 5.00)
].copy()

progress(f"\nRunners selected: {len(subset):,}", indent=1)
progress(f"Percentage of all runners: {(len(subset)/len(df)*100):.1f}%", indent=1)

if len(subset) == 0:
    progress("No runners met the filter criteria!", indent=1)
    conn.close()
    exit(1)

# Get actual race results from GreyhoundEntries
progress("\nLooking up actual race results...", indent=1)

results_query = """
SELECT 
    ge.GreyhoundID,
    g.GreyhoundName,
    rm.MeetingDate,
    r.RaceNumber,
    (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as IsWinner
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
"""

results_df = pd.read_sql_query(results_query, conn)
conn.close()

results_df['MeetingDate'] = pd.to_datetime(results_df['MeetingDate']).dt.date

# Join predicted bets with actual results
subset['MeetingDate'] = pd.to_datetime(subset['MeetingDate']).dt.date

# Merge on GreyhoundName, MeetingDate, RaceNumber
merged = subset.merge(
    results_df,
    on=['GreyhoundName', 'MeetingDate', 'RaceNumber'],
    how='left'
)

# Count wins and losses
wins = merged['IsWinner'].fillna(0).astype(int).sum()
bets = len(merged)
strike_rate = (wins / bets * 100) if bets > 0 else 0
avg_odds = merged['Odds'].mean()
total_return = (wins * avg_odds) + (bets - wins) * 0.0
roi = ((total_return - bets) / bets * 100) if bets > 0 else 0

progress("\n" + "="*100)
progress("RESULTS - WEIGHTED MODEL (Score >= 0.6, Odds $1.50-$5.00)")
progress("="*100)
progress(f"Total Bets: {bets:,}", indent=1)
progress(f"Total Wins: {wins:,}", indent=1)
progress(f"Strike Rate: {strike_rate:.1f}%", indent=1)
progress(f"Average Odds: ${avg_odds:.2f}", indent=1)
progress(f"Total Stake: ${bets:,.0f}", indent=1)
progress(f"Total Return: ${total_return:,.0f}", indent=1)
progress(f"Profit: ${total_return - bets:,.0f}", indent=1)
progress(f"ROI: {roi:+.1f}%", indent=1)

# Test high confidence only (score >= 0.75)
progress("\n" + "="*100)
progress("TESTING HIGH CONFIDENCE ONLY (Score >= 0.75, Odds $1.50-$5.00)")
progress("="*100)

high_conf = df[
    (df['WeightedScore'] >= 0.75) &
    (df['Odds'] >= 1.50) &
    (df['Odds'] <= 5.00)
].copy()

progress(f"\nRunners selected: {len(high_conf):,}", indent=1)

if len(high_conf) > 0:
    merged_hc = high_conf.merge(
        results_df,
        on=['GreyhoundName', 'MeetingDate', 'RaceNumber'],
        how='left'
    )
    
    wins_hc = merged_hc['IsWinner'].fillna(0).astype(int).sum()
    bets_hc = len(merged_hc)
    strike_rate_hc = (wins_hc / bets_hc * 100) if bets_hc > 0 else 0
    avg_odds_hc = merged_hc['Odds'].mean()
    total_return_hc = (wins_hc * avg_odds_hc) + (bets_hc - wins_hc) * 0.0
    roi_hc = ((total_return_hc - bets_hc) / bets_hc * 100) if bets_hc > 0 else 0
    
    progress(f"Total Bets: {bets_hc:,}", indent=1)
    progress(f"Total Wins: {wins_hc:,}", indent=1)
    progress(f"Strike Rate: {strike_rate_hc:.1f}%", indent=1)
    progress(f"Average Odds: ${avg_odds_hc:.2f}", indent=1)
    progress(f"ROI: {roi_hc:+.1f}%", indent=1)
else:
    progress("No runners at score >= 0.75!", indent=1)
    roi_hc = 0
    roi = 0

progress("\n" + "="*100)
progress("VERDICT")
progress("="*100)
progress(f"All Qualified (0.60+): ROI = {roi:+.1f}%", indent=1)
if len(high_conf) > 0:
    progress(f"High Confidence (0.75+): ROI = {roi_hc:+.1f}%", indent=1)
    if roi > roi_hc:
        progress(f"BETTER TO USE ALL QUALIFIED (difference: {roi - roi_hc:+.1f}pp)", indent=1)
    else:
        progress(f"HIGH CONFIDENCE IS BETTER (difference: {roi_hc - roi:+.1f}pp)", indent=1)
