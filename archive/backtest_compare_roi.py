"""
BACKTEST: Fast Check - Did the +13% ROI Finding Hold in 2025?
Simple direct approach: calculate historical pace, test on 2025 data
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

def progress(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

progress("="*100)
progress("BACKTEST: Investigating -3.19% vs +13.00% ROI Discrepancy")
progress("="*100)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

progress("\nSTEP 1: Get ALL race results to calculate historical pace")
query = """
SELECT ge.GreyhoundID, ge.Position, ge.StartingPrice,
       rm.MeetingDate, ge.FinishTimeBenchmarkLengths,
       rm.MeetingAvgBenchmarkLengths
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE ge.Position IS NOT NULL AND ge.Position NOT IN ('DNF', 'SCR')
  AND ge.FinishTimeBenchmarkLengths IS NOT NULL
  AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
  AND ge.StartingPrice IS NOT NULL
ORDER BY ge.GreyhoundID, rm.MeetingDate
"""

all_races = pd.read_sql_query(query, conn)
all_races['MeetingDate'] = pd.to_datetime(all_races['MeetingDate'])
all_races['Pace'] = all_races['FinishTimeBenchmarkLengths'] + all_races['MeetingAvgBenchmarkLengths']
progress(f"Loaded {len(all_races):,} total races")

progress("\nSTEP 2: Build historical pace database")
# Group by dog, sort by date
pace_db = {}
for dog_id in all_races['GreyhoundID'].unique():
    dog_data = all_races[all_races['GreyhoundID'] == dog_id].sort_values('MeetingDate')
    pace_db[dog_id] = list(zip(dog_data['MeetingDate'], dog_data['Pace']))

progress(f"Built database for {len(pace_db):,} dogs")

progress("\nSTEP 3: Get 2025 races and calculate their historical pace")
races_2025_query = """
SELECT ge.GreyhoundID, ge.Position, ge.StartingPrice,
       rm.MeetingDate, t.TrackName
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= '2025-01-01'
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
  AND ge.StartingPrice IS NOT NULL
"""

races_2025 = pd.read_sql_query(races_2025_query, conn)
races_2025['MeetingDate'] = pd.to_datetime(races_2025['MeetingDate'])
races_2025['IsWinner'] = (races_2025['Position'] == '1').astype(int)
races_2025['StartingPrice'] = pd.to_numeric(races_2025['StartingPrice'], errors='coerce')

progress(f"Loaded {len(races_2025):,} 2025 races")

progress("\nSTEP 4: Calculate historical pace for each 2025 race")
hist_paces = []
for _, row in races_2025.iterrows():
    dog_id = row['GreyhoundID']
    race_date = row['MeetingDate']
    
    if dog_id not in pace_db:
        hist_paces.append(np.nan)
        continue
    
    # Get all races BEFORE this race
    before = [p for d, p in pace_db[dog_id] if d < race_date]
    
    if len(before) == 0:
        hist_paces.append(np.nan)
    else:
        # Average of last 5 races
        hist_paces.append(np.mean(before[-5:]))

races_2025['HistoricalPace'] = hist_paces
races_2025_clean = races_2025.dropna(subset=['HistoricalPace', 'StartingPrice'])

progress(f"Calculated pace for {len(races_2025_clean):,} 2025 races with full data")

# Filter to Australian tracks
excluded = ['Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North',
           'Launceston', 'Hobart', 'Devonport']
races_2025_clean = races_2025_clean[~races_2025_clean['TrackName'].isin(excluded)]

progress(f"After filtering: {len(races_2025_clean):,} Australian races")

# Test different thresholds and odds ranges
progress("\nSTEP 5: Testing different pace thresholds on $1.50-$2.00 odds")
progress("="*100)

for threshold in [0.0, 0.25, 0.5, 1.0]:
    subset = races_2025_clean[
        (races_2025_clean['HistoricalPace'] >= threshold) &
        (races_2025_clean['StartingPrice'] >= 1.50) &
        (races_2025_clean['StartingPrice'] <= 2.00)
    ]
    
    if len(subset) < 10:
        continue
    
    wins = subset['IsWinner'].sum()
    bets = len(subset)
    strike = (wins / bets * 100)
    stake = bets * 1.0
    ret = (subset[subset['IsWinner'] == 1]['StartingPrice'].sum())
    roi = ((ret - stake) / stake * 100) if stake > 0 else 0
    profit = ret - stake
    
    print(f"Pace >= {threshold:.2f}: {bets:,} bets, {strike:.2f}% strike, ${profit:+.2f} profit, ROI {roi:+.2f}%")

conn.close()

progress("\n" + "="*100)
progress("ANALYSIS SUMMARY")
progress("="*100)
progress("""
DISCREPANCY EXPLANATION:

Original +13% ROI claim was based on:
  - test_pace_predictiveness.py
  - Used ROW_NUMBER() window function to select last 5 races
  - Tested on ALL historical data (not just 2025)
  
Backtest showing -3.19% ROI was:
  - backtest_pace_optimized.py  
  - Also calculated last 5 races correctly
  - But ONLY on 2025 data
  
The difference means:
  1. The +13% edge may not have held into 2025
  2. Market conditions changed (dogs/odds mix changed)
  3. Or the 2025 sample is too small to validate
  
NEXT STEP: Run backtest on ALL data (not just 2025) to see if
the +13% edge was real historically or just appeared randomly.
""")

progress(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
progress("="*100)
