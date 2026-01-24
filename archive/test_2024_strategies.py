"""Test different criteria combinations on 2024-2025 data"""
import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('greyhound_racing.db')

query = '''
SELECT 
    ge.GreyhoundID, ge.RaceID, ge.FirstSplitPosition, ge.Box,
    ge.Position, ge.StartingPrice, ge.CareerPrizeMoney,
    rm.MeetingDate
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
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['CareerPrizeMoney'] = pd.to_numeric(df['CareerPrizeMoney'], errors='coerce').fillna(0)
df['Box'] = pd.to_numeric(df['Box'], errors='coerce')
df['FirstSplitPosition'] = pd.to_numeric(df['FirstSplitPosition'], errors='coerce')
df['Win'] = (df['Position'] == '1') | (df['Position'] == 1)

df = df.sort_values(['GreyhoundID', 'MeetingDate', 'RaceID'])
df['CumCount'] = df.groupby('GreyhoundID').cumcount()
df['CumSplit'] = df.groupby('GreyhoundID')['FirstSplitPosition'].cumsum().shift(1)
df['HistAvgSplit'] = df['CumSplit'] / df['CumCount']
df['CumWins'] = df.groupby('GreyhoundID')['Win'].cumsum().shift(1).fillna(0)
df['HistWinRate'] = np.where(df['CumCount'] > 0, df['CumWins'] / df['CumCount'], 0)

box_adj = {1: -0.3, 2: -0.2, 3: -0.1, 4: 0, 5: 0, 6: 0.1, 7: 0.2, 8: 0.3}
df['BoxAdj'] = df['Box'].map(box_adj).fillna(0)
df['PredSplit'] = df['HistAvgSplit'] + df['BoxAdj']

df = df[df['CumCount'] >= 5].copy()
df = df.dropna(subset=['HistAvgSplit'])

df['PIRRank'] = df.groupby('RaceID')['PredSplit'].rank(method='min')
df['IsPIRLeader'] = df['PIRRank'] == 1

print('2024-2025 ONLY: Testing different criteria combinations')
print('='*70)

def test_strat(label, mask, odds_min=10, odds_max=50):
    s = df[mask & (df['StartingPrice'] >= odds_min) & (df['StartingPrice'] <= odds_max)]
    if len(s) < 20:
        return
    bets = len(s)
    wins = s['Win'].sum()
    returns = (s[s['Win']]['StartingPrice'] - 1).sum()
    roi = (returns - bets) / bets * 100
    days = len(s['MeetingDate'].unique())
    daily = bets / days if days > 0 else 0
    print(f'{label}: {bets} bets, {wins/bets*100:.1f}% wins, ROI: {roi:+.1f}%, {daily:.1f}/day')

# PIR leader only
test_strat('PIR leader only', df['IsPIRLeader'])

# PIR + high WR
test_strat('PIR + WR >= 30%', df['IsPIRLeader'] & (df['HistWinRate'] >= 0.30))
test_strat('PIR + WR >= 25%', df['IsPIRLeader'] & (df['HistWinRate'] >= 0.25))
test_strat('PIR + WR >= 20%', df['IsPIRLeader'] & (df['HistWinRate'] >= 0.20))

# PIR + high money
test_strat('PIR + Money >= 30k', df['IsPIRLeader'] & (df['CareerPrizeMoney'] >= 30000))
test_strat('PIR + Money >= 20k', df['IsPIRLeader'] & (df['CareerPrizeMoney'] >= 20000))
test_strat('PIR + Money >= 15k', df['IsPIRLeader'] & (df['CareerPrizeMoney'] >= 15000))

# Full combo 
test_strat('PIR + WR30 + M30k', df['IsPIRLeader'] & (df['HistWinRate'] >= 0.30) & (df['CareerPrizeMoney'] >= 30000))
test_strat('PIR + WR25 + M20k', df['IsPIRLeader'] & (df['HistWinRate'] >= 0.25) & (df['CareerPrizeMoney'] >= 20000))
test_strat('PIR + WR20 + M15k', df['IsPIRLeader'] & (df['HistWinRate'] >= 0.20) & (df['CareerPrizeMoney'] >= 15000))

print('')
print('Expanding odds range to 5-50:')
test_strat('PIR + WR30 + M30k', df['IsPIRLeader'] & (df['HistWinRate'] >= 0.30) & (df['CareerPrizeMoney'] >= 30000), 5, 50)
test_strat('PIR + WR25 + M20k', df['IsPIRLeader'] & (df['HistWinRate'] >= 0.25) & (df['CareerPrizeMoney'] >= 20000), 5, 50)
test_strat('PIR + WR20 + M15k', df['IsPIRLeader'] & (df['HistWinRate'] >= 0.20) & (df['CareerPrizeMoney'] >= 15000), 5, 50)

print('')
print('Just longshot range 10-50:')
test_strat('PIR only', df['IsPIRLeader'], 10, 50)
test_strat('PIR + WR >= 15%', df['IsPIRLeader'] & (df['HistWinRate'] >= 0.15), 10, 50)
test_strat('PIR + Money >= 10k', df['IsPIRLeader'] & (df['CareerPrizeMoney'] >= 10000), 10, 50)
test_strat('PIR + WR15 + M10k', df['IsPIRLeader'] & (df['HistWinRate'] >= 0.15) & (df['CareerPrizeMoney'] >= 10000), 10, 50)

conn.close()
