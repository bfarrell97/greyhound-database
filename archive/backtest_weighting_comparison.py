"""
BACKTEST: Compare different model weightings
Tests multiple combinations of Pace vs Form weighting
to find the optimal mix
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
progress("WEIGHTING COMPARISON: Testing different Pace vs Form combinations")
progress("="*100)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Get all unique race dates in November 2025
query_dates = """
SELECT DISTINCT DATE(rm.MeetingDate) as RaceDate
FROM Races r
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate >= '2025-11-01' AND rm.MeetingDate < '2025-12-01'
ORDER BY RaceDate
"""

progress("\nGetting all race dates in November 2025...", indent=1)
dates_df = pd.read_sql_query(query_dates, conn)
race_dates = dates_df['RaceDate'].tolist()
progress(f"Found {len(race_dates)} unique race dates", indent=1)

# For each date, simulate what bets would be made
all_bets = []
total_processed = 0

progress("\nSimulating daily predictions and tracking results...", indent=1)
progress(f"Processing {len(race_dates)} dates (this will take several minutes)...\n", indent=1)

for idx, race_date in enumerate(race_dates, 1):
    # Show progress every 10 dates
    if idx % 10 == 0:
        progress(f"[{idx}/{len(race_dates)}] Processing {race_date}... ({total_processed} bets so far)", indent=1)
    
    try:
        # Query: Get all dogs racing on this date with their historical pace/form UP TO this date
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
              AND rm.MeetingDate < DATE('{race_date}')
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
              AND rm.MeetingDate < DATE('{race_date}')
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
        ),
        
        todays_races AS (
            SELECT 
                ge.GreyhoundID,
                g.GreyhoundName,
                ge.StartingPrice,
                (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as ActualWinner
            FROM GreyhoundEntries ge
            JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
            JOIN Races r ON ge.RaceID = r.RaceID
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            WHERE DATE(rm.MeetingDate) = DATE('{race_date}')
              AND ge.Position IS NOT NULL
              AND ge.Position NOT IN ('DNF', 'SCR')
              AND ge.StartingPrice IS NOT NULL
        )
        
        SELECT 
            tr.GreyhoundName,
            tr.StartingPrice,
            dpa.HistoricalPaceAvg,
            COALESCE(dfl.RecentWinRate, 0) as RecentWinRate,
            tr.ActualWinner,
            DATE('{race_date}') as RaceDate
        FROM todays_races tr
        LEFT JOIN dog_pace_avg dpa ON tr.GreyhoundID = dpa.GreyhoundID
        LEFT JOIN dog_form_last_5 dfl ON tr.GreyhoundID = dfl.GreyhoundID
        """
        
        df = pd.read_sql_query(query, conn)
        
        if len(df) == 0:
            continue
        
        # Clean data
        df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
        df['HistoricalPaceAvg'] = pd.to_numeric(df['HistoricalPaceAvg'], errors='coerce')
        df['RecentWinRate'] = pd.to_numeric(df['RecentWinRate'], errors='coerce')
        df = df.dropna(subset=['StartingPrice', 'HistoricalPaceAvg'])
        
        if len(df) == 0:
            continue
        
        # Normalize scores
        pace_min = df['HistoricalPaceAvg'].min()
        pace_max = df['HistoricalPaceAvg'].max()
        
        if pace_max - pace_min == 0:
            df['PaceScore'] = 0.5
        else:
            df['PaceScore'] = (df['HistoricalPaceAvg'] - pace_min) / (pace_max - pace_min)
        
        df['FormScore'] = df['RecentWinRate'] / 100.0
        
        # Store all data for later processing
        df['RaceDate'] = race_date
        all_bets.append(df)
        total_processed += len(df)
    
    except Exception as e:
        progress(f"Error on {race_date}: {str(e)}", indent=1)
        continue

if len(all_bets) > 0:
    all_bets_df = pd.concat(all_bets, ignore_index=True)
else:
    progress("No bets generated!", indent=1)
    conn.close()
    exit(1)

conn.close()

progress(f"\nDone! Total predictions generated: {len(all_bets_df):,}", indent=1)

# ============================================================================
# TEST DIFFERENT WEIGHTINGS
# ============================================================================

weightings = [
    (0.50, 0.50, "50% Pace + 50% Form"),
    (0.60, 0.40, "60% Pace + 40% Form"),
    (0.70, 0.30, "70% Pace + 30% Form"),
    (0.80, 0.20, "80% Pace + 20% Form"),
    (0.90, 0.10, "90% Pace + 10% Form"),
    (1.00, 0.00, "100% Pace"),
]

thresholds = [0.60, 0.70, 0.75, 0.80]

results = []

progress("\n" + "="*100)
progress("TESTING DIFFERENT WEIGHTINGS AND THRESHOLDS")
progress("="*100)

for pace_weight, form_weight, label in weightings:
    progress(f"\n{label}:", indent=1)
    
    # Calculate weighted score
    all_bets_df['WeightedScore'] = (all_bets_df['PaceScore'] * pace_weight) + (all_bets_df['FormScore'] * form_weight)
    
    for threshold in thresholds:
        # Filter
        subset = all_bets_df[
            (all_bets_df['WeightedScore'] >= threshold) &
            (all_bets_df['StartingPrice'] >= 1.50) &
            (all_bets_df['StartingPrice'] <= 5.00)
        ].copy()
        
        if len(subset) == 0:
            continue
        
        wins = subset['ActualWinner'].astype(int).sum()
        bets = len(subset)
        strike_rate = (wins / bets * 100) if bets > 0 else 0
        avg_odds = subset['StartingPrice'].mean()
        total_return = (wins * avg_odds) + (bets - wins) * 0.0
        roi = ((total_return - bets) / bets * 100) if bets > 0 else 0
        
        results.append({
            'Weighting': label,
            'Threshold': threshold,
            'Bets': bets,
            'Wins': wins,
            'Strike': strike_rate,
            'AvgOdds': avg_odds,
            'ROI': roi
        })
        
        progress(f"  Threshold {threshold:.2f}: {bets:5} bets, {strike_rate:5.1f}% strike, ROI {roi:+6.1f}%", indent=2)

# ============================================================================
# SUMMARY TABLE
# ============================================================================

progress("\n" + "="*100)
progress("SUMMARY: BEST RESULTS BY WEIGHTING")
progress("="*100)

results_df = pd.DataFrame(results)

for pace_weight, form_weight, label in weightings:
    weighting_results = results_df[results_df['Weighting'] == label].sort_values('ROI', ascending=False)
    if len(weighting_results) > 0:
        best = weighting_results.iloc[0]
        progress(f"{label:25} | Best at {best['Threshold']:.2f}: {best['ROI']:+6.1f}% ROI ({best['Bets']:5} bets, {best['Strike']:5.1f}% strike)", indent=1)

# Find overall best
progress("\n" + "="*100)
progress("OVERALL BEST RESULTS")
progress("="*100)

best_overall = results_df.loc[results_df['ROI'].idxmax()]
progress(f"Best Configuration:", indent=1)
progress(f"  Weighting: {best_overall['Weighting']}", indent=2)
progress(f"  Threshold: {best_overall['Threshold']:.2f}", indent=2)
progress(f"  Bets: {best_overall['Bets']:,}", indent=2)
progress(f"  Wins: {best_overall['Wins']:,}", indent=2)
progress(f"  Strike Rate: {best_overall['Strike']:.1f}%", indent=2)
progress(f"  Average Odds: ${best_overall['AvgOdds']:.2f}", indent=2)
progress(f"  ROI: {best_overall['ROI']:+.1f}%", indent=2)

# Show top 5 configurations
progress("\n" + "="*100)
progress("TOP 5 CONFIGURATIONS")
progress("="*100)

top_5 = results_df.nlargest(5, 'ROI')
for i, row in enumerate(top_5.itertuples(), 1):
    progress(f"{i}. {row.Weighting:25} @ {row.Threshold:.2f}: {row.ROI:+6.1f}% ROI ({row.Bets:5} bets, {row.Strike:5.1f}% strike)", indent=1)

progress("\n" + "="*100)
