"""
BACKTEST: Exact Replication of Original +13% ROI Validation
Uses the SAME methodology as test_pace_predictiveness.py
Compares 2025 data to validate if the +13% claim still holds
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
progress("BACKTEST: COMPARING ORIGINAL VALIDATION TO 2025 DATA")
progress("="*100)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# ============================================================================
# REPLICATE ORIGINAL VALIDATION METHODOLOGY
# ============================================================================
progress("\nREPLICATING ORIGINAL VALIDATION")
progress("Using EXACT same SQL as test_pace_predictiveness.py")

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

future_races AS (
    SELECT 
        ge.GreyhoundID,
        (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as IsWinner,
        ge.StartingPrice,
        dpa.HistoricalPaceAvg,
        rm.MeetingDate,
        t.TrackName,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
    FROM GreyhoundEntries ge
    JOIN dog_pace_avg dpa ON ge.GreyhoundID = dpa.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.Position IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
      AND ge.StartingPrice IS NOT NULL
      AND rm.MeetingDate >= '2025-01-01'
)

SELECT * FROM future_races WHERE RaceNum > 1
"""

progress("Loading data with original methodology...", indent=1)
start = time.time()
df = pd.read_sql_query(query, conn)
elapsed = time.time() - start
progress(f"Loaded {len(df):,} races in {elapsed:.1f}s", indent=1)

# Clean data
df['IsWinner'] = pd.to_numeric(df['IsWinner'], errors='coerce').fillna(0)
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['HistoricalPaceAvg'] = pd.to_numeric(df['HistoricalPaceAvg'], errors='coerce')

# Remove NZ/AU non-Australian tracks
excluded = ['Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North',
           'Launceston', 'Hobart', 'Devonport']
df = df[~df['TrackName'].isin(excluded)]

df = df.dropna(subset=['IsWinner', 'HistoricalPaceAvg', 'StartingPrice'])

progress(f"After cleaning: {len(df):,} races with complete data", indent=1)
progress(f"Overall win rate: {df['IsWinner'].mean()*100:.2f}%", indent=1)

# ============================================================================
# TEST: Pace >= 0.5 on $1.50-$2.00 odds (ORIGINAL SWEET SPOT)
# ============================================================================
progress("\nTEST: Historical Pace >= 0.5 on $1.50-$2.00 odds")

above_pace = df[df['HistoricalPaceAvg'] >= 0.5]
in_odds = above_pace[
    (above_pace['StartingPrice'] >= 1.50) &
    (above_pace['StartingPrice'] <= 2.00)
].copy()

if len(in_odds) > 0:
    wins = in_odds['IsWinner'].sum()
    total = len(in_odds)
    strike = (wins / total * 100)
    
    # ROI calculation
    stakes = total * 1.0
    returns = (in_odds[in_odds['IsWinner'] == 1]['StartingPrice'].sum())
    roi = ((returns - stakes) / stakes * 100) if stakes > 0 else 0
    profit = returns - stakes
    
    progress(f"Bets: {total:,}", indent=1)
    progress(f"Wins: {wins:,}", indent=1)
    progress(f"Strike: {strike:.2f}%", indent=1)
    progress(f"Stake: ${stakes:,.2f}", indent=1)
    progress(f"Returns: ${returns:,.2f}", indent=1)
    progress(f"Profit: ${profit:+,.2f}", indent=1)
    progress(f"ROI: {roi:+.2f}%", indent=1)
    
    if roi >= 10:
        progress(f"\nRESULT: POSITIVE ROI CONFIRMED ({roi:+.2f}%)", indent=1)
    else:
        progress(f"\nRESULT: ROI BELOW TARGET ({roi:+.2f}% vs +13% expected)", indent=1)

conn.close()

progress(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
progress("="*100)
