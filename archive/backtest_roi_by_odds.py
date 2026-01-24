"""
BACKTEST: ROI Analysis by Odds Ranges
Tests the optimal model (70% Pace + 30% Form @ 0.80 threshold)
across different odds profiles
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
progress("ROI ANALYSIS BY ODDS RANGES")
progress("="*100)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Get all unique race dates in 2025
query_dates = """
SELECT DISTINCT DATE(rm.MeetingDate) as RaceDate
FROM Races r
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate >= '2025-01-01'
ORDER BY RaceDate
"""

progress("\nGetting all race dates in 2025...", indent=1)
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
        
        # Calculate weighted score (70/30)
        df['WeightedScore'] = (df['PaceScore'] * 0.7) + (df['FormScore'] * 0.3)
        
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
# FILTER TO OPTIMAL CONFIGURATION
# ============================================================================

progress("\n" + "="*100)
progress("FILTERING TO OPTIMAL MODEL: 70% Pace + 30% Form @ 0.80 threshold")
progress("="*100)

# Filter to score >= 0.80
optimal = all_bets_df[
    (all_bets_df['WeightedScore'] >= 0.80) &
    (all_bets_df['StartingPrice'] >= 1.50) &
    (all_bets_df['StartingPrice'] <= 5.00)
].copy()

progress(f"Total bets qualifying: {len(optimal):,}", indent=1)

# ============================================================================
# ANALYZE BY ODDS RANGES
# ============================================================================

progress("\n" + "="*100)
progress("ROI BREAKDOWN BY ODDS RANGES")
progress("="*100)

odds_ranges = [
    (1.50, 2.00, "$1.50-$2.00"),
    (2.00, 2.50, "$2.00-$2.50"),
    (2.50, 3.00, "$2.50-$3.00"),
    (3.00, 3.50, "$3.00-$3.50"),
    (3.50, 4.00, "$3.50-$4.00"),
    (4.00, 5.00, "$4.00-$5.00"),
]

results = []

for min_odds, max_odds, label in odds_ranges:
    subset = optimal[
        (optimal['StartingPrice'] >= min_odds) &
        (optimal['StartingPrice'] < max_odds)
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
        'Range': label,
        'Bets': bets,
        'Wins': wins,
        'Strike': strike_rate,
        'AvgOdds': avg_odds,
        'Return': total_return,
        'ROI': roi
    })
    
    progress(f"{label:15} | {bets:4} bets | {wins:3} wins | {strike_rate:5.1f}% strike | Avg ${avg_odds:5.2f} | ROI {roi:+6.1f}%", indent=1)

results_df = pd.DataFrame(results)

# ============================================================================
# CUMULATIVE ANALYSIS
# ============================================================================

progress("\n" + "="*100)
progress("CUMULATIVE ANALYSIS: ROI by Progressive Odds Ranges")
progress("="*100)

cumulative_ranges = [
    (1.50, 2.00, "$1.50-$2.00"),
    (1.50, 2.50, "$1.50-$2.50"),
    (1.50, 3.00, "$1.50-$3.00"),
    (1.50, 3.50, "$1.50-$3.50"),
    (1.50, 4.00, "$1.50-$4.00"),
    (1.50, 5.00, "$1.50-$5.00"),
]

for min_odds, max_odds, label in cumulative_ranges:
    subset = optimal[
        (optimal['StartingPrice'] >= min_odds) &
        (optimal['StartingPrice'] <= max_odds)
    ].copy()
    
    if len(subset) == 0:
        continue
    
    wins = subset['ActualWinner'].astype(int).sum()
    bets = len(subset)
    strike_rate = (wins / bets * 100) if bets > 0 else 0
    avg_odds = subset['StartingPrice'].mean()
    total_return = (wins * avg_odds) + (bets - wins) * 0.0
    roi = ((total_return - bets) / bets * 100) if bets > 0 else 0
    
    progress(f"{label:15} | {bets:5} bets | {strike_rate:5.1f}% strike | Avg ${avg_odds:5.2f} | ROI {roi:+6.1f}%", indent=1)

# ============================================================================
# SUMMARY
# ============================================================================

progress("\n" + "="*100)
progress("SUMMARY")
progress("="*100)

overall_wins = optimal['ActualWinner'].astype(int).sum()
overall_bets = len(optimal)
overall_strike = (overall_wins / overall_bets * 100) if overall_bets > 0 else 0
overall_avg_odds = optimal['StartingPrice'].mean()
overall_return = (overall_wins * overall_avg_odds) + (overall_bets - overall_wins) * 0.0
overall_roi = ((overall_return - overall_bets) / overall_bets * 100) if overall_bets > 0 else 0

progress(f"Overall Performance (70% Pace + 30% Form @ 0.80):", indent=1)
progress(f"  Total Bets: {overall_bets:,}", indent=2)
progress(f"  Total Wins: {overall_wins:,}", indent=2)
progress(f"  Strike Rate: {overall_strike:.1f}%", indent=2)
progress(f"  Average Odds: ${overall_avg_odds:.2f}", indent=2)
progress(f"  Total Return: ${overall_return:,.0f}", indent=2)
progress(f"  Total ROI: {overall_roi:+.1f}%", indent=2)

# Find best odds range
best_range = results_df.loc[results_df['ROI'].idxmax()]
progress(f"\nBest Odds Range:", indent=1)
progress(f"  Range: {best_range['Range']}", indent=2)
progress(f"  ROI: {best_range['ROI']:+.1f}%", indent=2)
progress(f"  Bets: {best_range['Bets']}", indent=2)
progress(f"  Strike: {best_range['Strike']:.1f}%", indent=2)

progress("\n" + "="*100)
