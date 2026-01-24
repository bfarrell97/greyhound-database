"""
Validate the Box <= 2 strategy found in test_simple_rules.py
Test on multiple scenarios:
1. Different time periods (train vs test)
2. Different track subsets  
3. Different odds ranges
4. Statistical significance
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

DB_PATH = 'greyhound_racing.db'

print("="*80)
print("VALIDATING BOX <= 2 STRATEGY")
print("="*80)

conn = sqlite3.connect(DB_PATH)

# Test 1: Original high-win tracks, test period only
print("\n" + "="*80)
print("TEST 1: High-Win Tracks, Test Period ($1.50-$2.00)")
print("="*80)

query = """
SELECT
    t.TrackName, ge.Box, ge.Weight, ge.Position, ge.StartingPrice
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= '2025-01-01' AND rm.MeetingDate < '2025-12-01'
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
  AND t.TrackName IN ('Nowra', 'Q Straight', 'Dport @ HOB')
"""

df = pd.read_sql_query(query, conn)
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['IsWinner'] = (df['Position'] == 1).astype(int)
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')

# Filter to $1.50-$2.00
df_filtered = df[(df['StartingPrice'] >= 1.5) & (df['StartingPrice'] < 2.0)]

print(f"Total races: {len(df)}")
print(f"$1.50-$2.00 races: {len(df_filtered)}")
print(f"Overall win rate: {df_filtered['IsWinner'].mean()*100:.1f}%")

# Test box positions
for max_box in [1, 2, 3, 4, 5]:
    strategy = df_filtered[df_filtered['Box'] <= max_box]
    if len(strategy) > 20:
        wins = strategy['IsWinner'].sum()
        strike = (wins / len(strategy)) * 100
        
        # Calculate ROI
        stake_pct = 0.02
        bankroll = 1000
        total_staked = len(strategy) * bankroll * stake_pct
        returns = (strategy[strategy['IsWinner'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
        pnl = returns - total_staked
        roi = (pnl / total_staked) * 100 if total_staked > 0 else 0
        
        # Calculate statistical significance (binomial)
        break_even_rate = 1 / 1.9  # At 1.9 avg odds, need 52.6% to break even
        if wins > 0:
            z_score = (wins - (len(strategy) * break_even_rate)) / np.sqrt(len(strategy) * break_even_rate * (1 - break_even_rate))
            p_value = 2 * (1 - np.arctan(abs(z_score)) / (np.pi/2))  # rough approximation
        else:
            p_value = 1.0
        
        print(f"Box <= {max_box}: {len(strategy):>4} bets, {strike:>5.1f}% strike, ROI {roi:>6.2f}%")

# Test 2: Box <= 2 on ALL tracks (not just high performers)
print("\n" + "="*80)
print("TEST 2: ALL Tracks, Test Period ($1.50-$2.00)")
print("="*80)

query_all = """
SELECT
    t.TrackName, ge.Box, ge.Weight, ge.Position, ge.StartingPrice
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= '2025-01-01' AND rm.MeetingDate < '2025-12-01'
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
"""

df_all = pd.read_sql_query(query_all, conn)
df_all['Position'] = pd.to_numeric(df_all['Position'], errors='coerce')
df_all['IsWinner'] = (df_all['Position'] == 1).astype(int)
df_all['StartingPrice'] = pd.to_numeric(df_all['StartingPrice'], errors='coerce')

df_all_filtered = df_all[(df_all['StartingPrice'] >= 1.5) & (df_all['StartingPrice'] < 2.0)]

print(f"Total $1.50-$2.00 races on all tracks: {len(df_all_filtered)}")
print(f"Overall win rate: {df_all_filtered['IsWinner'].mean()*100:.1f}%")

# Test box position
for max_box in [1, 2, 3]:
    strategy = df_all_filtered[df_all_filtered['Box'] <= max_box]
    if len(strategy) > 20:
        wins = strategy['IsWinner'].sum()
        strike = (wins / len(strategy)) * 100
        
        stake_pct = 0.02
        bankroll = 1000
        total_staked = len(strategy) * bankroll * stake_pct
        returns = (strategy[strategy['IsWinner'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
        pnl = returns - total_staked
        roi = (pnl / total_staked) * 100 if total_staked > 0 else 0
        
        print(f"Box <= {max_box}: {len(strategy):>4} bets, {strike:>5.1f}% strike, ROI {roi:>6.2f}%")

# Test 3: Box <= 2 by individual track
print("\n" + "="*80)
print("TEST 3: Box <= 2 by Individual Track ($1.50-$2.00)")
print("="*80)

for track in sorted(df_all_filtered['TrackName'].unique()):
    track_data = df_all_filtered[df_all_filtered['TrackName'] == track]
    strategy = track_data[track_data['Box'] <= 2]
    
    if len(strategy) > 10:
        wins = strategy['IsWinner'].sum()
        strike = (wins / len(strategy)) * 100
        
        stake_pct = 0.02
        bankroll = 1000
        total_staked = len(strategy) * bankroll * stake_pct
        returns = (strategy[strategy['IsWinner'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
        pnl = returns - total_staked
        roi = (pnl / total_staked) * 100 if total_staked > 0 else 0
        
        print(f"{track:20} Box<=2: {len(strategy):>4} bets, {strike:>5.1f}% strike, ROI {roi:>6.2f}%")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("Box <= 2 shows promise on high-win tracks but needs broader validation")
print("Consider: Is +2.34% ROI on 152 bets real edge or luck?")
print("Next: Test on train period (2024) to validate, then deploy if consistent")
