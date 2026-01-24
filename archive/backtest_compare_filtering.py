"""
BACKTEST COMPARISON: With vs Without Outlier Filtering
Tests the impact of removing anomalous benchmark values
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
progress("BACKTEST COMPARISON: Outlier Filtering Impact Analysis")
progress("="*100)

DB_PATH = 'greyhound_racing.db'

def run_backtest(use_outlier_filter=False, description=""):
    """Run backtest with or without outlier filtering"""
    
    conn = sqlite3.connect(DB_PATH)
    
    # Build WHERE clause conditionally
    outlier_clause = "AND ge.FinishTimeBenchmarkLengths BETWEEN -20.52 AND 9.18" if use_outlier_filter else ""
    
    query = f"""
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
      {outlier_clause}
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
WHERE drf.RaceNum > 1
  AND drf.MeetingDate >= '2025-01-01'
"""
    
    progress(f"\n{description}...", indent=1)
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
    
    # Calculate scores
    pace_min = df['HistoricalPaceAvg'].min()
    pace_max = df['HistoricalPaceAvg'].max()
    if pace_max - pace_min == 0:
        df['PaceScore'] = 0.5
    else:
        df['PaceScore'] = (df['HistoricalPaceAvg'] - pace_min) / (pace_max - pace_min)
    
    df['FormScore'] = df['RecentWinRate'] / 100.0
    df['WeightedScore'] = (df['PaceScore'] * 0.7) + (df['FormScore'] * 0.3)
    
    # Test at threshold 0.6 on $1.50-$2.00 odds
    subset = df[
        (df['WeightedScore'] >= 0.6) &
        (df['StartingPrice'] >= 1.50) &
        (df['StartingPrice'] <= 2.00)
    ].copy()
    
    wins = subset['IsWinner'].sum()
    bets = len(subset)
    
    if bets == 0:
        strike_rate = 0
        roi = 0
        avg_odds = 0
    else:
        strike_rate = wins / bets * 100
        avg_odds = subset['StartingPrice'].mean()
        total_return = (wins * avg_odds) + (bets - wins) * 0.0
        roi = ((total_return - bets) / bets) * 100
    
    return {
        'description': description,
        'bets': bets,
        'wins': wins,
        'strike_rate': strike_rate,
        'avg_odds': avg_odds,
        'roi': roi,
        'df': df,
        'subset': subset
    }

# Run both versions
progress("\n" + "="*100)
without_filter = run_backtest(use_outlier_filter=False, description="LOADING DATA: Without Outlier Filter")
with_filter = run_backtest(use_outlier_filter=True, description="LOADING DATA: With Outlier Filter")

# Display comparison
progress("\n" + "="*100)
progress("COMPARISON RESULTS")
progress("="*100)

progress(f"\nWITHOUT OUTLIER FILTER:", indent=1)
progress(f"  Bets: {without_filter['bets']:,}", indent=2)
progress(f"  Wins: {without_filter['wins']:,}", indent=2)
progress(f"  Strike Rate: {without_filter['strike_rate']:.1f}%", indent=2)
progress(f"  Average Odds: ${without_filter['avg_odds']:.2f}", indent=2)
progress(f"  ROI: {without_filter['roi']:+.1f}%", indent=2)

progress(f"\nWITH OUTLIER FILTER:", indent=1)
progress(f"  Bets: {with_filter['bets']:,}", indent=2)
progress(f"  Wins: {with_filter['wins']:,}", indent=2)
progress(f"  Strike Rate: {with_filter['strike_rate']:.1f}%", indent=2)
progress(f"  Average Odds: ${with_filter['avg_odds']:.2f}", indent=2)
progress(f"  ROI: {with_filter['roi']:+.1f}%", indent=2)

progress(f"\nIMPACT OF FILTERING:", indent=1)
bet_change = with_filter['bets'] - without_filter['bets']
roi_change = with_filter['roi'] - without_filter['roi']
strike_change = with_filter['strike_rate'] - without_filter['strike_rate']

progress(f"  Bet Change: {bet_change:+,} ({(bet_change/without_filter['bets']*100):+.1f}%)", indent=2)
progress(f"  Strike Rate Change: {strike_change:+.1f}pp", indent=2)
progress(f"  ROI Change: {roi_change:+.1f}pp", indent=2)

if roi_change > 0:
    progress(f"  VERDICT: Filtering IMPROVES ROI by {roi_change:.1f}pp", indent=2)
elif roi_change < 0:
    progress(f"  VERDICT: Filtering REDUCES ROI by {abs(roi_change):.1f}pp", indent=2)
else:
    progress(f"  VERDICT: Filtering has no effect on ROI", indent=2)

progress("\n" + "="*100)

# Analysis: Compare score distributions
progress("\nSCORE DISTRIBUTION ANALYSIS:", indent=1)

progress(f"\n  Without Filter - PaceScore distribution:", indent=2)
progress(f"    Min: {without_filter['df']['PaceScore'].min():.3f}", indent=3)
progress(f"    Q1:  {without_filter['df']['PaceScore'].quantile(0.25):.3f}", indent=3)
progress(f"    Mean: {without_filter['df']['PaceScore'].mean():.3f}", indent=3)
progress(f"    Q3:  {without_filter['df']['PaceScore'].quantile(0.75):.3f}", indent=3)
progress(f"    Max: {without_filter['df']['PaceScore'].max():.3f}", indent=3)

progress(f"\n  With Filter - PaceScore distribution:", indent=2)
progress(f"    Min: {with_filter['df']['PaceScore'].min():.3f}", indent=3)
progress(f"    Q1:  {with_filter['df']['PaceScore'].quantile(0.25):.3f}", indent=3)
progress(f"    Mean: {with_filter['df']['PaceScore'].mean():.3f}", indent=3)
progress(f"    Q3:  {with_filter['df']['PaceScore'].quantile(0.75):.3f}", indent=3)
progress(f"    Max: {with_filter['df']['PaceScore'].max():.3f}", indent=3)

progress(f"\n  Without Filter - WeightedScore >= 0.6:", indent=2)
progress(f"    Count: {len(without_filter['subset']):,}", indent=3)
progress(f"    Mean Strike: {without_filter['subset']['IsWinner'].mean()*100:.1f}%", indent=3)

progress(f"\n  With Filter - WeightedScore >= 0.6:", indent=2)
progress(f"    Count: {len(with_filter['subset']):,}", indent=3)
progress(f"    Mean Strike: {with_filter['subset']['IsWinner'].mean()*100:.1f}%", indent=3)

progress("\n" + "="*100)
