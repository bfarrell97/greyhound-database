"""Test if adding pace metric improves the PIR + Money strategy"""
import sqlite3
import pandas as pd
import numpy as np

print('='*70)
print('TESTING: Does adding PACE improve PIR + Money strategy?')
print('='*70)
print()

conn = sqlite3.connect('greyhound_racing.db')

query = '''
SELECT 
    ge.GreyhoundID, ge.RaceID, ge.FirstSplitPosition, ge.Box,
    ge.Position, ge.StartingPrice, ge.CareerPrizeMoney,
    ge.FinishTimeBenchmarkLengths,
    rm.MeetingDate, rm.MeetingAvgBenchmarkLengths
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE ge.Position IS NOT NULL
  AND ge.StartingPrice IS NOT NULL
  AND ge.FirstSplitPosition IS NOT NULL
  AND ge.FirstSplitPosition != ''
  AND rm.MeetingDate >= '2024-01-01'
ORDER BY ge.GreyhoundID, rm.MeetingDate, r.RaceID
'''

df = pd.read_sql_query(query, conn)
print(f'Loaded {len(df)} rows')

df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['CareerPrizeMoney'] = pd.to_numeric(df['CareerPrizeMoney'], errors='coerce').fillna(0)
df['Box'] = pd.to_numeric(df['Box'], errors='coerce')
df['FirstSplitPosition'] = pd.to_numeric(df['FirstSplitPosition'], errors='coerce')
df['FinishTimeBenchmarkLengths'] = pd.to_numeric(df['FinishTimeBenchmarkLengths'], errors='coerce')
df['MeetingAvgBenchmarkLengths'] = pd.to_numeric(df['MeetingAvgBenchmarkLengths'], errors='coerce').fillna(0)
df['Win'] = (df['Position'] == '1') | (df['Position'] == 1)

# Calculate total pace (adjusted for meeting)
df['TotalPace'] = df['FinishTimeBenchmarkLengths'] + df['MeetingAvgBenchmarkLengths']

df = df.sort_values(['GreyhoundID', 'MeetingDate', 'RaceID'])

# Cumulative historical metrics
df['CumCount'] = df.groupby('GreyhoundID').cumcount()

# PIR (split position)
df['CumSplit'] = df.groupby('GreyhoundID')['FirstSplitPosition'].cumsum().shift(1)
df['HistAvgSplit'] = df['CumSplit'] / df['CumCount']

# Pace (finish benchmark)
df['CumPace'] = df.groupby('GreyhoundID')['TotalPace'].cumsum().shift(1)
df['HistAvgPace'] = df['CumPace'] / df['CumCount']

# Box adjustment for PIR
box_adj = {1: -0.3, 2: -0.2, 3: -0.1, 4: 0, 5: 0, 6: 0.1, 7: 0.2, 8: 0.3}
df['BoxAdj'] = df['Box'].map(box_adj).fillna(0)
df['PredSplit'] = df['HistAvgSplit'] + df['BoxAdj']

# Filter to dogs with enough history
df = df[df['CumCount'] >= 5].copy()
df = df.dropna(subset=['HistAvgSplit'])

# Rank within race
df['PIRRank'] = df.groupby('RaceID')['PredSplit'].rank(method='min')
df['IsPIRLeader'] = df['PIRRank'] == 1

# Pace rank (lower is better - faster)
df['PaceRank'] = df.groupby('RaceID')['HistAvgPace'].rank(method='min', ascending=True)
df['IsPaceLeader'] = df['PaceRank'] == 1

# Odds filter
ODDS_MIN = 15
ODDS_MAX = 50
MONEY_THRESH = 30000

odds_mask = (df['StartingPrice'] >= ODDS_MIN) & (df['StartingPrice'] <= ODDS_MAX)
money_mask = df['CareerPrizeMoney'] >= MONEY_THRESH

def test_strategy(name, mask):
    s = df[mask & odds_mask]
    if len(s) < 20:
        print(f'{name}: Not enough bets')
        return
    bets = len(s)
    wins = s['Win'].sum()
    returns = (s[s['Win']]['StartingPrice'] - 1).sum()
    roi = (returns - bets) / bets * 100
    days = len(s['MeetingDate'].unique())
    daily = bets / days
    print(f'{name}:')
    print(f'  {bets} bets, {wins} wins ({wins/bets*100:.1f}%), ROI: {roi:+.1f}%, {daily:.1f}/day')
    print()

print()
print('='*70)
print('BASELINE: PIR Leader + Money >= $30k')
print('='*70)
test_strategy('PIR + Money', df['IsPIRLeader'] & money_mask)

print('='*70)
print('ADDING PACE CONDITIONS')
print('='*70)

# Test various pace additions
test_strategy('PIR + Money + PaceLeader', 
              df['IsPIRLeader'] & money_mask & df['IsPaceLeader'])

test_strategy('PIR + Money + PaceTop3', 
              df['IsPIRLeader'] & money_mask & (df['PaceRank'] <= 3))

test_strategy('PIR + Money + PaceTop5', 
              df['IsPIRLeader'] & money_mask & (df['PaceRank'] <= 5))

# What about pace instead of PIR?
print('='*70)
print('COMPARING: Pace vs PIR')
print('='*70)

test_strategy('PaceLeader + Money (no PIR)', 
              df['IsPaceLeader'] & money_mask)

test_strategy('PaceTop3 + Money (no PIR)', 
              (df['PaceRank'] <= 3) & money_mask)

# Combined scoring approach
print('='*70)
print('COMBINED SCORING: PIR + Pace')
print('='*70)

# Normalize and combine
df['PIRScore'] = 1 - (df['PredSplit'] - df['PredSplit'].min()) / (df['PredSplit'].max() - df['PredSplit'].min())
df['PaceScore'] = 1 - (df['HistAvgPace'] - df['HistAvgPace'].min()) / (df['HistAvgPace'].max() - df['HistAvgPace'].min())

# Fill NaN pace scores with 0.5 (neutral)
df['PaceScore'] = df['PaceScore'].fillna(0.5)

for pir_wt in [0.7, 0.5, 0.3]:
    pace_wt = 1 - pir_wt
    df['CombinedScore'] = (df['PIRScore'] * pir_wt) + (df['PaceScore'] * pace_wt)
    df['CombinedRank'] = df.groupby('RaceID')['CombinedScore'].rank(method='min', ascending=False)
    
    test_strategy(f'Combined ({int(pir_wt*100)}% PIR + {int(pace_wt*100)}% Pace) Top1 + Money',
                  (df['CombinedRank'] == 1) & money_mask)

# What if we require BOTH to be leaders?
print('='*70)
print('STRICT: Must be leader in BOTH PIR and Pace')
print('='*70)

test_strategy('PIR Leader AND Pace Leader + Money',
              df['IsPIRLeader'] & df['IsPaceLeader'] & money_mask)

# Without money filter
test_strategy('PIR Leader AND Pace Leader (no money filter)',
              df['IsPIRLeader'] & df['IsPaceLeader'])

conn.close()
print('DONE')
