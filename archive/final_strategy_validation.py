"""Final validation of PIR + Money strategy for 2024-2025"""
import sqlite3
import pandas as pd
import numpy as np
import sys

def fp(msg):
    print(msg)
    sys.stdout.flush()

fp('='*70)
fp('FINAL STRATEGY VALIDATION: PIR Leader + High Career Money')
fp('='*70)
fp('')

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
fp(f'Loaded {len(df)} rows for 2024-2025')

df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['CareerPrizeMoney'] = pd.to_numeric(df['CareerPrizeMoney'], errors='coerce').fillna(0)
df['Box'] = pd.to_numeric(df['Box'], errors='coerce')
df['FirstSplitPosition'] = pd.to_numeric(df['FirstSplitPosition'], errors='coerce')
df['Win'] = (df['Position'] == '1') | (df['Position'] == 1)
df['Month'] = pd.to_datetime(df['MeetingDate']).dt.to_period('M')

df = df.sort_values(['GreyhoundID', 'MeetingDate', 'RaceID'])
df['CumCount'] = df.groupby('GreyhoundID').cumcount()
df['CumSplit'] = df.groupby('GreyhoundID')['FirstSplitPosition'].cumsum().shift(1)
df['HistAvgSplit'] = df['CumSplit'] / df['CumCount']

box_adj = {1: -0.3, 2: -0.2, 3: -0.1, 4: 0, 5: 0, 6: 0.1, 7: 0.2, 8: 0.3}
df['BoxAdj'] = df['Box'].map(box_adj).fillna(0)
df['PredSplit'] = df['HistAvgSplit'] + df['BoxAdj']

df = df[df['CumCount'] >= 5].copy()
df = df.dropna(subset=['HistAvgSplit'])

df['PIRRank'] = df.groupby('RaceID')['PredSplit'].rank(method='min')
df['IsPIRLeader'] = df['PIRRank'] == 1

# Best strategy - higher ROI version
MONEY_THRESH = 30000
ODDS_MIN = 15
ODDS_MAX = 50

mask = (df['IsPIRLeader'] & 
        (df['CareerPrizeMoney'] >= MONEY_THRESH) & 
        (df['StartingPrice'] >= ODDS_MIN) & 
        (df['StartingPrice'] <= ODDS_MAX))

subset = df[mask].copy()
fp(f'Strategy: PIR Leader + Money >= ${MONEY_THRESH:,} @ ${ODDS_MIN}-${ODDS_MAX}')
fp(f'Total qualifying bets: {len(subset)}')
fp('')

# Overall stats
bets = len(subset)
wins = subset['Win'].sum()
returns = (subset[subset['Win']]['StartingPrice'] - 1).sum()
roi = (returns - bets) / bets * 100
days = len(subset['MeetingDate'].unique())
daily = bets / days

fp('='*70)
fp('OVERALL PERFORMANCE')
fp('='*70)
fp(f'Bets: {bets}')
fp(f'Wins: {wins} ({wins/bets*100:.1f}%)')
fp(f'ROI: {roi:+.1f}%')
fp(f'Days with bets: {days}')
fp(f'Average bets/day: {daily:.1f}')
fp('')

# Month by month
fp('='*70)
fp('MONTH BY MONTH')
fp('='*70)

cumulative_profit = 0
for month in sorted(subset['Month'].unique()):
    m = subset[subset['Month'] == month]
    m_bets = len(m)
    m_wins = m['Win'].sum()
    m_returns = (m[m['Win']]['StartingPrice'] - 1).sum()
    m_profit = m_returns - m_bets
    m_roi = m_profit / m_bets * 100 if m_bets > 0 else 0
    cumulative_profit += m_profit
    fp(f'{month}: {m_bets:4d} bets, {m_wins:3d} wins ({m_wins/m_bets*100:5.1f}%), ROI: {m_roi:+6.1f}%, Cumulative: {cumulative_profit:+.1f} units')

# Shuffle test - compare to random selection at same odds
fp('')
fp('='*70)
fp('STATISTICAL VALIDATION (Monte Carlo)')
fp('='*70)

same_odds = df[(df['StartingPrice'] >= ODDS_MIN) & (df['StartingPrice'] <= ODDS_MAX)]
fp(f'Pool size (all dogs at ${ODDS_MIN}-${ODDS_MAX}): {len(same_odds)}')

np.random.seed(42)
rand_rois = []
for i in range(100):
    sample = same_odds.sample(n=min(len(subset), len(same_odds)), replace=True)
    s_wins = sample['Win'].sum()
    s_returns = (sample[sample['Win']]['StartingPrice'] - 1).sum()
    s_roi = (s_returns - len(sample)) / len(sample) * 100
    rand_rois.append(s_roi)

avg_rand = np.mean(rand_rois)
std_rand = np.std(rand_rois)
z = (roi - avg_rand) / std_rand if std_rand > 0 else 0

fp('')
fp(f'Our strategy ROI: {roi:+.1f}%')
fp(f'Random baseline: {avg_rand:+.1f}% (std: {std_rand:.1f}%)')
fp(f'Edge over random: {roi - avg_rand:+.1f}%')
fp(f'Z-score: {z:.2f}')
fp('')

if z > 3:
    fp('*** HIGHLY SIGNIFICANT (p < 0.001) ***')
elif z > 2:
    fp('*** STATISTICALLY SIGNIFICANT (p < 0.05) ***')
elif z > 1.5:
    fp('Marginally significant')
else:
    fp('Not statistically significant')

# Practical metrics
fp('')
fp('='*70)
fp('PRACTICAL DEPLOYMENT')
fp('='*70)
fp(f'Expected bets per day: {daily:.1f}')
fp(f'Expected ROI: {roi:+.1f}%')
fp(f'If betting $10/bet: ${daily*10:.0f}/day wagered')
fp(f'Expected daily profit: ${daily*10*roi/100:.2f}')
fp('')

conn.close()
fp('VALIDATION COMPLETE')
