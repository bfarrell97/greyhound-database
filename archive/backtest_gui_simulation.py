"""
PROPER BACKTEST: Simulate GUI predictions on historical races
For each historical race date, calculate what the GUI would have predicted,
then check if those predictions actually won
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
progress("PROPER BACKTEST: GUI Simulation on Historical Races")
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
    )
    
    SELECT 
        ge.GreyhoundID,
        g.GreyhoundName,
        ge.StartingPrice,
        (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as ActualWinner,
        COALESCE(dfl.RecentWins, 0) as RecentWins,
        COALESCE(dfl.RecentRaces, 0) as RecentRaces,
        COALESCE(dfl.RecentWinRate, 0) as RecentWinRate,
        COALESCE(dpa.HistoricalPaceAvg, 0) as HistoricalPaceAvg,
        DATE(rm.MeetingDate) as RaceDate
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    LEFT JOIN dog_pace_avg dpa ON ge.GreyhoundID = dpa.GreyhoundID
    LEFT JOIN dog_form_last_5 dfl ON ge.GreyhoundID = dfl.GreyhoundID
    WHERE DATE(rm.MeetingDate) = DATE('{race_date}')
      AND ge.Position IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
      AND ge.StartingPrice IS NOT NULL
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        
        if len(df) == 0:
            continue
        
        # Clean data
        df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
        df['HistoricalPaceAvg'] = pd.to_numeric(df['HistoricalPaceAvg'], errors='coerce')
        df['RecentWinRate'] = pd.to_numeric(df['RecentWinRate'], errors='coerce')
        df['ActualWinner'] = pd.to_numeric(df['ActualWinner'], errors='coerce')
        
        # Remove rows without historical pace data
        df = df.dropna(subset=['HistoricalPaceAvg', 'StartingPrice'])
        
        if len(df) == 0:
            continue
        
        # Calculate weighted scores (same as GUI)
        pace_min = df['HistoricalPaceAvg'].min()
        pace_max = df['HistoricalPaceAvg'].max()
        
        if pace_max - pace_min == 0:
            df['PaceScore'] = 0.5
        else:
            df['PaceScore'] = (df['HistoricalPaceAvg'] - pace_min) / (pace_max - pace_min)
        
        df['FormScore'] = df['RecentWinRate'] / 100.0
        df['WeightedScore'] = (df['PaceScore'] * 0.7) + (df['FormScore'] * 0.3)
        
        # Filter to score >= 0.80 and odds $1.50-$5.00
        filtered = df[
            (df['WeightedScore'] >= 0.80) &
            (df['StartingPrice'] >= 1.50) &
            (df['StartingPrice'] <= 5.00)
        ].copy()
        
        if len(filtered) > 0:
            filtered['Score'] = filtered['WeightedScore']
            filtered['Odds'] = filtered['StartingPrice']
            filtered['IsWinner'] = filtered['ActualWinner']
            all_bets.append(filtered[['GreyhoundName', 'RaceDate', 'Odds', 'Score', 'IsWinner']])
            total_processed += len(filtered)
    
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
# BACKTEST RESULTS
# ============================================================================
progress("\n" + "="*100)
progress("BACKTEST RESULTS: 70% Pace + 30% Form (Score >= 0.80, Odds $1.50-$5.00)")
progress("="*100)

wins = all_bets_df['IsWinner'].astype(int).sum()
bets = len(all_bets_df)
strike_rate = (wins / bets * 100) if bets > 0 else 0
avg_odds = all_bets_df['Odds'].mean()
total_return = (wins * avg_odds) + (bets - wins) * 0.0
roi = ((total_return - bets) / bets * 100) if bets > 0 else 0

progress(f"Total Bets: {bets:,}", indent=1)
progress(f"Total Wins: {wins:,}", indent=1)
progress(f"Strike Rate: {strike_rate:.1f}%", indent=1)
progress(f"Average Odds: ${avg_odds:.2f}", indent=1)
progress(f"Total Stake: ${bets:,.0f}", indent=1)
progress(f"Total Return: ${total_return:,.0f}", indent=1)
progress(f"Profit: ${total_return - bets:,.0f}", indent=1)
progress(f"ROI: {roi:+.1f}%", indent=1)

# Price range breakdown
progress("\n" + "="*100)
progress("PRICE RANGE BREAKDOWN (Score >= 0.80)")
progress("="*100)

price_ranges = [
    (1.50, 2.00, "$1.50-$2.00"),
    (2.00, 3.00, "$2.00-$3.00"),
    (3.00, 4.00, "$3.00-$4.00"),
    (4.00, 5.00, "$4.00-$5.00"),
]

for min_price, max_price, label in price_ranges:
    price_band = all_bets_df[(all_bets_df['Odds'] >= min_price) & (all_bets_df['Odds'] < max_price)]
    if len(price_band) > 0:
        wins_p = price_band['IsWinner'].astype(int).sum()
        bets_p = len(price_band)
        strike_p = (wins_p / bets_p * 100) if bets_p > 0 else 0
        avg_odds_p = price_band['Odds'].mean()
        return_p = (wins_p * avg_odds_p)
        roi_p = ((return_p - bets_p) / bets_p * 100) if bets_p > 0 else 0
        
        progress(f"{label:15} | {bets_p:4} bets | {strike_p:5.1f}% strike | Avg ${avg_odds_p:.2f} | ROI {roi_p:+6.1f}%", indent=1)
    else:
        progress(f"{label:15} | No bets", indent=1)

# Test lower threshold for comparison
progress("\n" + "="*100)
progress("COMPARISON: Score >= 0.75 vs Score >= 0.80")
progress("="*100)

high_conf = all_bets_df[all_bets_df['Score'] >= 0.75]

if len(high_conf) > 0:
    wins_hc = high_conf['IsWinner'].astype(int).sum()
    bets_hc = len(high_conf)
    strike_rate_hc = (wins_hc / bets_hc * 100) if bets_hc > 0 else 0
    avg_odds_hc = high_conf['Odds'].mean()
    total_return_hc = (wins_hc * avg_odds_hc) + (bets_hc - wins_hc) * 0.0
    roi_hc = ((total_return_hc - bets_hc) / bets_hc * 100) if bets_hc > 0 else 0
    
    progress(f"Score >= 0.75: {bets_hc:,} bets, {strike_rate_hc:.1f}% strike, ROI {roi_hc:+.1f}%", indent=1)
    progress(f"Score >= 0.80: {bets:,} bets, {strike_rate:.1f}% strike, ROI {roi:+.1f}%", indent=1)
    
    if roi > roi_hc:
        diff = roi - roi_hc
        progress(f"Score >= 0.80 is BETTER (better by {diff:+.1f}pp ROI)", indent=1)
    else:
        diff = roi_hc - roi
        progress(f"Score >= 0.75 is BETTER (better by {diff:+.1f}pp ROI)", indent=1)
else:
    progress("No bets at score >= 0.75!", indent=1)

# Breakdown by score bands
progress("\n" + "="*100)
progress("BREAKDOWN BY SCORE BANDS")
progress("="*100)

bands = [
    (0.60, 0.65, "0.60-0.65"),
    (0.65, 0.70, "0.65-0.70"),
    (0.70, 0.75, "0.70-0.75"),
    (0.75, 0.80, "0.75-0.80"),
    (0.80, 1.00, "0.80+"),
]

for min_score, max_score, label in bands:
    band = all_bets_df[(all_bets_df['Score'] >= min_score) & (all_bets_df['Score'] < max_score)]
    if len(band) > 0:
        wins_b = band['IsWinner'].astype(int).sum()
        bets_b = len(band)
        strike_b = (wins_b / bets_b * 100) if bets_b > 0 else 0
        avg_odds_b = band['Odds'].mean()
        return_b = (wins_b * avg_odds_b)
        roi_b = ((return_b - bets_b) / bets_b * 100) if bets_b > 0 else 0
        
        progress(f"{label:12} | {bets_b:4} bets | {strike_b:5.1f}% strike | ROI {roi_b:+6.1f}%", indent=1)

progress("\n" + "="*100)
