"""
TEST: Weighted Model combining Historical Pace + Recent Form
Tests different weight combinations to find optimal ROI
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

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

progress("="*100)
progress("TEST: WEIGHTED MODEL - Historical Pace + Recent Form")
progress("="*100)

# Get dogs with both metrics
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
        COUNT(DISTINCT CASE WHEN RaceNum <= 5 THEN MeetingDate END) as RacesUsed
    FROM dog_pace_history
    GROUP BY GreyhoundID
    HAVING RacesUsed >= 5
),

dog_recent_form AS (
    SELECT 
        ge.GreyhoundID,
        g.GreyhoundName,
        (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as IsWinner,
        ge.StartingPrice,
        rm.MeetingDate,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
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
        GreyhoundName,
        COUNT(*) as RecentRaces,
        SUM(CASE WHEN IsWinner = 1 THEN 1 ELSE 0 END) as RecentWins,
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
WHERE drf.RaceNum > 1
"""

progress("Loading data with both metrics...", indent=1)
df = pd.read_sql_query(query, conn)
conn.close()

# Clean data
df['IsWinner'] = pd.to_numeric(df['IsWinner'], errors='coerce').fillna(0)
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['HistoricalPaceAvg'] = pd.to_numeric(df['HistoricalPaceAvg'], errors='coerce')
df['RecentWins'] = pd.to_numeric(df['RecentWins'], errors='coerce').fillna(0)
df['RecentRaces'] = pd.to_numeric(df['RecentRaces'], errors='coerce').fillna(0)
df['RecentWinRate'] = pd.to_numeric(df['RecentWinRate'], errors='coerce').fillna(0)

# Remove NZ/AU non-Australian tracks
excluded = ['Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North',
           'Launceston', 'Hobart', 'Devonport']
df = df[~df['GreyhoundName'].isin(excluded)]

df = df.dropna(subset=['IsWinner', 'HistoricalPaceAvg', 'StartingPrice'])

progress(f"Loaded {len(df):,} races", indent=1)

# Normalize metrics to 0-1 scale for weighting
df['PaceScore'] = (df['HistoricalPaceAvg'] - df['HistoricalPaceAvg'].min()) / (df['HistoricalPaceAvg'].max() - df['HistoricalPaceAvg'].min())
df['FormScore'] = df['RecentWinRate'] / 100.0  # Already 0-100

progress("\n" + "="*100)
progress("TESTING DIFFERENT WEIGHT COMBINATIONS")
progress("="*100 + "\n")

# Test different weight combinations
test_weights = [
    (1.0, 0.0, "Pure Historical Pace (Original Model)"),
    (0.7, 0.3, "70% Pace, 30% Form"),
    (0.5, 0.5, "50% Pace, 50% Form (Equal Weight)"),
    (0.3, 0.7, "30% Pace, 70% Form"),
    (0.0, 1.0, "Pure Recent Form (No Pace)"),
]

results_summary = []

for pace_weight, form_weight, description in test_weights:
    progress(f"\nTesting: {description}")
    progress("-" * 100, indent=1)
    
    # Calculate weighted score
    df['WeightedScore'] = (df['PaceScore'] * pace_weight) + (df['FormScore'] * form_weight)
    
    # Test at different thresholds on $1.50-$2.00 odds
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        subset = df[
            (df['WeightedScore'] >= threshold) &
            (df['StartingPrice'] >= 1.50) &
            (df['StartingPrice'] <= 2.00)
        ].copy()
        
        if len(subset) == 0:
            continue
        
        wins = subset['IsWinner'].sum()
        bets = len(subset)
        strike_rate = wins / bets * 100
        
        # Calculate ROI
        # At $1.50-$2.00, average odds ~$1.75, payout = $1.75 per dollar
        avg_odds = subset['StartingPrice'].mean()
        total_return = (wins * avg_odds) + (bets - wins) * 0.0  # Lose stake on losses
        roi = ((total_return - bets) / bets) * 100
        
        results_summary.append({
            'Model': description,
            'Pace%': int(pace_weight * 100),
            'Form%': int(form_weight * 100),
            'Threshold': threshold,
            'Bets': bets,
            'Wins': wins,
            'Strike%': strike_rate,
            'AvgOdds': f"${avg_odds:.2f}",
            'ROI%': roi,
        })
        
        status = "[GOOD]" if roi > 10 else "[OK]" if roi > 0 else "[BAD]"
        progress(f"Threshold {threshold:.1f}: {bets:4} bets, {wins:3} wins ({strike_rate:5.1f}%), AvgOdds ${avg_odds:.2f}, ROI {roi:+6.1f}% {status}", indent=2)

# Find best combinations
progress("\n" + "="*100)
progress("SUMMARY: Best Performing Models (ROI > 10%)")
progress("="*100)

results_df = pd.DataFrame(results_summary)
best = results_df[results_df['ROI%'] > 10].sort_values('ROI%', ascending=False)

if len(best) > 0:
    print("\n" + best.to_string(index=False))
else:
    progress("No combinations achieved ROI > 10%", indent=1)
    # Show top 10
    progress("Top 10 Results:")
    top10 = results_df.sort_values('ROI%', ascending=False).head(10)
    print("\n" + top10.to_string(index=False))

progress("\n" + "="*100)
progress("RECOMMENDATION")
progress("="*100)

best_single = results_df.loc[results_df['ROI%'].idxmax()]
progress(f"Best Model: {best_single['Model']}", indent=1)
progress(f"Threshold: {best_single['Threshold']}", indent=1)
progress(f"Expected: {best_single['Bets']} bets, {best_single['Strike%']:.1f}% strike, {best_single['ROI%']:+.1f}% ROI", indent=1)
